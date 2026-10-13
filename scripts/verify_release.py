"""Actual recomputation from saved raw outputs; ZIP-only fresh runs."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sparseradiomap.versioned import prepare_dataset,atomic_json,open_store,Store,task,array_sha,file_sha,utc
from sparseradiomap.data import read_manifest,load_image,write_csv
from sparseradiomap.core import metrics

def recompute(root,frame,label):
    rows=[];max_error=0.;json_error=0.
    for _,row in frame.iterrows():
        img=load_image(root,row)
        with np.load(root/row.prediction_path) as f:raw=f['raw']
        with np.load(root/row.mask_path) as f:obs,hole=f['observed'],f['hole']
        assert raw.shape==img['y'].shape and np.isfinite(raw).all()
        assert not (obs&~img['valid']).any()
        score=metrics(raw,img['y'],img['valid'],obs,hole)
        record_path=(root/'results/runs/main'/f'{row.run_id}.json') if label=='baseline_v01' else (root/'results/test_v02/runs'/f'{row.run_id}.json')
        record=json.loads(record_path.read_text())
        for key,v in score.items():
            if key not in frame.columns:continue
            old=row[key]
            if np.isnan(v) and pd.isna(old):continue
            max_error=max(max_error,abs(float(v)-float(old)))
            json_error=max(json_error,abs(float(v)-float(record[key])))
        rows.append(dict(run_id=row.run_id,version=label,map_id=row.map_id,tx_id=row.tx_id,
                         method=row.method,fraction=row.fraction,mechanism=row.mechanism,seed=row.seed,**score))
    return pd.DataFrame(rows),dict(records=len(rows),max_csv_difference=max_error,max_json_difference=json_error)

def verify_all(out,baseline):
    out=Path(out);baseline=Path(baseline)
    store=open_store(out,'test_v02');new=store.export();checked={}
    for label,root,frame in [('baseline_v01',baseline,pd.read_csv(baseline/'results/main.csv')),('test_v02',out,new)]:
        if label=='baseline_v01' and not all((root/r.prediction_path).exists() and (root/r.mask_path).exists() for r in frame.itertuples()):
            checked[label]={'status':'reference_arrays_not_bundled','note':'Baseline recomputation was executed in the complete local workspace; public package excludes arrays. No new verification claimed.'}
            continue
        recalculated,info=recompute(root,frame,label)
        if max(info['max_csv_difference'],info['max_json_difference'])>1e-5:raise ValueError('Score inconsistency')
        recalculated.to_csv(out/'reports'/f'{label}_recomputed.csv',index=False);checked[label]=info
    for _,g in new.groupby(['map_id','tx_id','fraction','mechanism','seed']):
        assert g.mask_sha256.nunique()==1 and set(g.method)=={'idw','smooth','lowrank','mlp'}
    for _,g in new.groupby(['map_id','tx_id','fraction','seed','method']):assert g.n_observed.nunique()==1
    selection=json.loads((out/'configs/selected_v02.json').read_text())
    assert selection['scientific_source_sha256']==store.code
    for m,p in selection['methods'].items():
        assert all(json.loads(v)==p for v in new[new.method==m].params_json)
    atomic_json(out/'reports/score_verification.json',dict(checks=checked,
                paired_masks=216,equal_observation_counts=True,main_recomputed_from_raw=True,
                devices='CPU only actually tested',completed_utc=utc()))
    print('Metrics independently recomputed:',checked,flush=True)

def fresh(out,archive,split,baseline,force_cpu=False):
    out=Path(out);baseline=Path(baseline);archive=Path(archive)
    if out.exists() and any(out.iterdir()):raise ValueError('fresh requires a genuinely empty output directory')
    c=json.loads((baseline/'configs/experiment.json').read_text())
    if split=='dev':
        row=next(r for r in read_manifest(baseline) if r['split']=='dev')
        params={m:v[0] for m,v in c['candidates'].items()}
        reference_path=baseline/'results/smoke.csv'
        refroot=baseline
    else:
        refroot=baseline/'experiments/v02'
        params=json.loads((refroot/'configs/selected_v02.json').read_text())['methods']
        row=next(r for r in read_manifest(baseline) if r['split']=='test')
        reference_path=refroot/'results/test_v02/results.csv'
    expected=json.loads((baseline/'reports/download.json').read_text())['sha256']
    if force_cpu:
        params={m:p.copy() for m,p in params.items()}
        for method in ['lowrank','mlp']:params[method]['device']='cpu'
    rows=prepare_dataset(out,archive,[row],expected)
    cfg=dict(protocol=c,methods=params,source='ZIP-only genuinely clean first run',split=split)
    version='fresh_'+split+'_v02';atomic_json(out/'configs'/f'{version}.json',cfg)
    tasks=[task(out,c,rows[0],m,p,.05,'random',11) for m,p in params.items()]
    store=Store(out,version,cfg,tasks);counts=store.run()
    assert counts['refitted']==4 and counts['cached_verified']==0
    result=store.export();comparisons=[]
    reference=pd.read_csv(reference_path)  # reference metrics only after all fresh fits
    for _,r in result.iterrows():
        ref=reference[(reference.map_id==r.map_id)&(reference.tx_id==r.tx_id)&(reference.fraction==.05)&(reference.mechanism=='random')&(reference.seed==11)&(reference.method==r.method)].iloc[0]
        if not (refroot/ref.prediction_path).exists():
            comparisons.append(dict(method=r.method,reference_available=False));continue
        with np.load(out/r.prediction_path) as f:pred=f['raw']
        with np.load(refroot/ref.prediction_path) as f:target=f['raw']
        with np.load(out/r.mask_path) as f:mask=f['observed'];hole=f['hole']
        with np.load(refroot/ref.mask_path) as f:equal=np.array_equal(mask,f['observed']) and np.array_equal(hole,f['hole'])
        delta=float(np.max(np.abs(pred-target)));score_delta=abs(float(r.hidden_rmse_db)-float(ref.hidden_rmse_db))
        assert equal and delta<1e-5 and score_delta<1e-4
        comparisons.append(dict(method=r.method,reference_available=True,masks_equal=equal,
                                max_prediction_abs_difference=delta,hidden_rmse_db_difference=score_delta))
    import importlib.metadata as metadata
    packages={d.metadata['Name']:d.version for d in metadata.distributions()}
    atomic_json(out/'reports/fresh_verification.json',dict(refitted=4,cached=0,split=split,
                input_only='SHA-256 verified original ZIP; no borrowed processed arrays, masks, predictions or run JSON',
                comparisons=comparisons,package_versions=packages,completed_utc=utc(),device='cpu'))
    print('Fresh run:',comparisons,flush=True)
