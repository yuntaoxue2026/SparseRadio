"""Development/validation-only budget study and source-bound selection."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from .data import read_manifest,write_csv
from .training import train_trajectory
from .versioned import Store,task,digest,atomic_json,scientific_hash,utc

CHECKPOINTS=[200,1000,3000]

def planned(root,config,base,split,version,dev_regions=2):
    rows=[r for r in read_manifest(root) if r['split']==split]
    seeds=config['seeds']
    if split=='dev':
        regions=list(dict.fromkeys(r['map_id'] for r in rows))[:dev_regions]
        rows=[r for r in rows if r['map_id'] in regions];seeds=[11]
    if split not in ['dev','val']:raise ValueError('Diagnostics cannot use test data')
    tasks=[]
    for r in rows:
        for fraction in config['fractions']:
            for mechanism in config['mechanisms']:
                for seed in seeds:
                    for method in ['lowrank','mlp']:
                        for step in CHECKPOINTS:
                            params=dict(base['methods'][method],steps=step)
                            tasks.append(task(root,config,r,method,params,fraction,mechanism,seed,checkpoint_step=step))
    study=dict(protocol=config,baseline_methods=base['methods'],checkpoints=CHECKPOINTS,
               split=split,seeds=seeds,dev_rule='First two development regions in original preselected manifest; all two Tx; seed 11',
               validation_rule='All six validation regions, both Tx, both mechanisms, all three fractions, seeds 11/22/33',
               logging_steps=[0,1,10,25,50,100,200,400,600,800,1000,1500,2000,2500,3000],
               metric_timing='After parameter update; minibatch loss explicitly pre-update',
               score='Mean region hidden RMSE; equal mechanisms and fractions, Tx/seeds within region',
               further_tuning_rule='After budget validation only: if normalized full-observed RMSE remains >=0.04 for every budget, inspect regularization/objective plateau on dev first. At most two additional regularization or LR candidates; ranks/architecture unchanged. No test-based tuning.')
    atomic_json(Path(root)/'configs'/f'{version}.json',study)
    return Store(root,version,study,tasks)

def trajectories(store):
    store.check_source();groups={}
    for ident,t in store.entries.items():
        variant=digest({k:v for k,v in t['params'].items() if k!='steps'})
        key=tuple(t[k] for k in ['map_id','tx_id','fraction','mechanism','seed','method'])+(variant,)
        groups.setdefault(key,[]).append((ident,t))
    counts=dict(trajectories_refitted=0,trajectories_cached_verified=0,checkpoints_created=0)
    for i,(key,group) in enumerate(groups.items()):
        variant=key[-1]
        cache=[store.cached(ident) is not None for ident,t in group]
        # Partial trajectory results are re-fit from step zero with the same RNG.
        if all(cache):counts['trajectories_cached_verified']+=1;continue
        t=group[0][1];image,obs,hole,_=store.masks(t)
        scoring=[]
        def score_callback(step,raw,info):
            from .core import metrics
            evaluation=metrics(raw,image['y'],image['valid'],obs,hole)
            scoring.append(dict(info,map_id=t['map_id'],tx_id=t['tx_id'],split=t['split'],
                          fraction=t['fraction'],mechanism=t['mechanism'],seed=t['seed'],method=t['method'],
                          candidate_id=variant,params_json=json.dumps({k:v for k,v in t['params'].items() if k!='steps'},sort_keys=True),
                          hidden_rmse=evaluation['hidden_rmse'],hidden_rmse_db=evaluation['hidden_rmse_db'],
                          hole_rmse_db=evaluation['hole_rmse_db'],clip_fraction_outdoor=evaluation['clip_fraction']))
        checkpoints=sorted({g['checkpoint_step'] for _,g in group})
        predictions,history=train_trajectory(t['method'],image['y'].shape,np.argwhere(obs),image['y'][obs],
                                             t['params'],t['mask_seed'],checkpoints,score_callback)
        trajectory_id=digest({'plan':store.plan_sha,'group':key})
        write_csv(store.directory/'trajectories'/f'{trajectory_id}.csv',scoring)
        for ident,checkpoint in group:
            step=checkpoint['checkpoint_step'];log=next(r for r in history if r['step']==step)
            store.save(ident,predictions[step],log['elapsed_seconds'],dict(trajectory_id=trajectory_id,
                       optimization_seconds=log['optimization_seconds'],metrics_timing=log['metrics_timing'],
                       data_fit_mse=log['data_fit_mse'],regularization=log['regularization'],total_objective=log['total_objective'],
                       observed_rmse_raw=log['observed_rmse_raw'],lr=log['lr'],device='cpu'))
            counts['checkpoints_created']+=1
        counts['trajectories_refitted']+=1
        if (i+1)%12==0:print(store.version,i+1,'/',len(groups),counts,flush=True)
    df=store.export()
    curves=pd.concat([pd.read_csv(p) for p in sorted((store.directory/'trajectories').glob('*.csv'))],ignore_index=True)
    if len(curves.groupby(['map_id','tx_id','fraction','mechanism','seed','method','candidate_id']))!=len(groups):raise ValueError('Missing/extra trajectories')
    curves.to_csv(store.directory/'training_curves.csv',index=False)
    atomic_json(store.directory/'invocation.json',dict(counts,completed_utc=utc()))
    if store.config['split']=='dev':
        times=df[df.checkpoint_step==3000].groupby('method').seconds.mean().to_dict()
        estimate=216*sum(times.values())
        atomic_json(store.root/'reports/diagnostic_budget.json',dict(development_trajectories=len(groups),
                   mean_3000_seconds=times,estimated_validation_seconds=estimate,
                   decision='Use all six validation regions and all three seeds; no size reduction',
                   estimate_scope='Includes diagnostic scoring/forward passes; excludes output I/O'))
    return df

def regularization_plan(root,config,base,validation_store,version='ridge_val_v02'):
    """Evidence gate defined before viewing the validation budget results.

    Only low-rank, two ridge values, at the best validation-selected budget.
    No architecture/rank/LR search and no holdout-based decisions.
    """
    df=validation_store.export();low=df[df.method=='lowrank']
    observed=low.groupby('checkpoint_step').observed_rmse_raw.mean()
    gate=bool((observed>=.04).all())
    scores=low.groupby(['checkpoint_step','map_id','fraction','mechanism']).hidden_rmse.mean().groupby('checkpoint_step').mean()
    selected_step=int(scores.sort_values().index[0])
    evidence=dict(gate='Low-rank mean full-observed raw RMSE >=0.04 at every budget',
                  triggered=gate,observed_rmse_by_step={str(k):v for k,v in observed.items()},
                  selected_step=selected_step,ridge_candidates=[.01,.1],rank_and_lr='Unchanged',
                  recorded_utc=utc(),source_validation_plan_sha256=validation_store.plan_sha)
    atomic_json(Path(root)/'reports/further_tuning_gate.json',evidence)
    if not gate:return None
    tasks=[]
    for r in read_manifest(root):
        if r['split']!='val':continue
        for fraction in config['fractions']:
            for mechanism in config['mechanisms']:
                for seed in config['seeds']:
                    for ridge in [.01,.1]:
                        params=dict(base['methods']['lowrank'],reg=ridge,steps=selected_step)
                        tasks.append(task(root,config,r,'lowrank',params,fraction,mechanism,seed,checkpoint_step=selected_step))
    study=dict(protocol=config,split='val',gate_evidence=evidence,checkpoints=[selected_step],
               score='Same balanced region hidden RMSE as budget study',
               logging_steps=[0,1,10,25,50,100,200,400,600,800,1000,1500,2000,2500,3000])
    atomic_json(Path(root)/'configs'/f'{version}.json',study)
    return Store(root,version,study,tasks)

def choose(root,validation_store,base,version='test_v02',additional_store=None):
    root=Path(root);df=validation_store.export()
    if set(df.split)!= {'val'}:raise ValueError('Budget selection must use validation only')
    if additional_store is not None:
        additional=additional_store.export()
        if set(additional.split)!={'val'}:raise ValueError('Further tuning cannot use holdout')
        df=pd.concat([df,additional],ignore_index=True)
    region=df.groupby(['method','params_json','checkpoint_step','map_id','fraction','mechanism'],as_index=False).hidden_rmse.mean()
    candidates=region.groupby(['method','params_json','checkpoint_step'],as_index=False).hidden_rmse.mean()
    candidates['rmse_db']=100*candidates.hidden_rmse
    methods={m:cfg.copy() for m,cfg in base['methods'].items()}
    for method,g in candidates.groupby('method'):
        chosen=g.sort_values(['hidden_rmse','checkpoint_step']).iloc[0]
        methods[method]=json.loads(chosen.params_json)
    write_csv(root/'reports/budget_candidate_scores.csv',candidates.to_dict('records'))
    saved=dict(version=version,methods=methods,scientific_source_sha256=scientific_hash(),
               validation_version=validation_store.version,validation_plan_sha256=validation_store.plan_sha,
               additional_validation_plan_sha256=additional_store.plan_sha if additional_store else None,
               candidates_sha256=digest(candidates.to_dict('records')),validation_records=len(df),
               selection_rule='Equal-weight mean region hidden RMSE across both mechanisms and all three fractions; average Tx/seed within each region/condition; tie selects fewer steps',
               unchanged_methods='IDW and smooth parameters frozen from baseline; not re-selected from seen holdout',
               holdout_status='Follow-up exploratory evaluation on the same previously viewed holdout',frozen_utc=utc())
    dest=root/'configs/selected_v02.json'
    if dest.exists():
        old=json.loads(dest.read_text())
        for k in ['scientific_source_sha256','validation_plan_sha256','methods','candidates_sha256']:
            if old[k]!=saved[k]:raise ValueError('Selection/source changed: create a new version')
        return old
    atomic_json(dest,saved);return saved

def main_store(root,config,selected,version='test_v02'):
    if selected['scientific_source_sha256']!=scientific_hash():raise ValueError('Source changed after configuration selection')
    cfg=dict(protocol=config,selection=selected)
    atomic_json(Path(root)/'configs'/f'{version}.json',cfg)
    tasks=[]
    for row in read_manifest(root):
        if row['split']!='test':continue
        for fraction in config['fractions']:
            for mechanism in config['mechanisms']:
                for seed in config['seeds']:
                    for method,params in selected['methods'].items():
                        tasks.append(task(root,config,row,method,params,fraction,mechanism,seed))
    return Store(root,version,cfg,tasks)
