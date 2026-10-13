"""Version-labelled region statistics, budget curves, paired comparisons and reports."""
from pathlib import Path
import itertools,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sparseradiomap.versioned import open_store,atomic_json,file_sha,utc
from sparseradiomap.data import read_manifest,load_image
METHODS=['idw','smooth','lowrank','mlp']
LABELS={'idw':'IDW','smooth':'Grid smooth','lowrank':'Low-rank','mlp':'Coordinate MLP'}

def table(df):
    def value(x):return f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(value(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))

def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix('.png'),dpi=160,bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight');plt.close(fig)

def region_summary(df,version,rng):
    fields=['hidden_mae_db','hidden_rmse_db','hidden_bias_db','observed_rmse_db','hole_rmse_db','gradient_rmse_db','seconds','clip_fraction','raw_hidden_rmse']
    region=df.groupby(['map_id','fraction','mechanism','method'],as_index=False)[fields].mean()
    region['version']=version;summary=[]
    for (fraction,mechanism,method),g in region.groupby(['fraction','mechanism','method']):
        v=g.hidden_rmse_db.to_numpy();draw=rng.integers(len(v),size=(2000,len(v)));boot=v[draw].mean(1)
        summary.append(dict(version=version,fraction=fraction,mechanism=mechanism,method=method,n_regions=len(v),
                       rmse_db=v.mean(),ci_low_db=np.quantile(boot,.025),ci_high_db=np.quantile(boot,.975),
                       region_sd_db=v.std(ddof=1),mae_db=g.hidden_mae_db.mean(),bias_db=g.hidden_bias_db.mean(),
                       observed_rmse_db=g.observed_rmse_db.mean(),hole_rmse_db=g.hole_rmse_db.mean(),
                       gradient_rmse_db=g.gradient_rmse_db.mean(),seconds=g.seconds.mean(),
                       clip_fraction=g.clip_fraction.mean(),raw_rmse_db=100*g.raw_hidden_rmse.mean()))
    return region,pd.DataFrame(summary)

def summarize(out,baseline):
    out=Path(out);baseline=Path(baseline);figs=out/'figures';reports=out/'reports';reports.mkdir(parents=True,exist_ok=True)
    new=open_store(out,'test_v02').export()
    if len(new)!=864 or set(new.split)!={'test'}:raise ValueError('Incomplete new test comparison')
    old=pd.read_csv(baseline/'results/main.csv')
    rng=np.random.default_rng(31415)
    regions={};summaries={}
    for label,df in [('baseline_v01',old),('test_v02',new)]:
        reg,s=region_summary(df,label,rng);regions[label]=reg;summaries[label]=s
        reg.to_csv(out/'results'/f'{label}_region_metrics.csv',index=False)
        s.to_csv(out/'results'/f'{label}_summary.csv',index=False)
    summary=pd.concat(list(summaries.values()),ignore_index=True)
    summary.to_csv(out/'results/version_comparison_summary.csv',index=False)
    paired=[]
    for fraction in [.01,.05,.1]:
        for mechanism in ['random','block']:
            for method in METHODS:
                a=regions['baseline_v01'];b=regions['test_v02']
                a=a[(a.fraction==fraction)&(a.mechanism==mechanism)&(a.method==method)].set_index('map_id')
                b=b[(b.fraction==fraction)&(b.mechanism==mechanism)&(b.method==method)].set_index('map_id').loc[a.index]
                delta=(b.hidden_rmse_db-a.hidden_rmse_db).to_numpy()
                boot=delta[rng.integers(6,size=(2000,6))].mean(1)
                paired.append(dict(fraction=fraction,mechanism=mechanism,method=method,n_regions=6,
                                   delta_new_minus_baseline_db=delta.mean(),ci_low_db=np.quantile(boot,.025),ci_high_db=np.quantile(boot,.975)))
    paired=pd.DataFrame(paired);paired.to_csv(out/'results/baseline_new_paired.csv',index=False)
    method_pairs=[]
    for (fraction,mechanism),g in regions['test_v02'].groupby(['fraction','mechanism']):
        wide=g.pivot(index='map_id',columns='method',values='hidden_rmse_db');draw=rng.integers(6,size=(2000,6))
        for a,b in itertools.combinations(METHODS,2):
            d=(wide[a]-wide[b]).to_numpy();boot=d[draw].mean(1)
            method_pairs.append(dict(version='test_v02',fraction=fraction,mechanism=mechanism,method_a=a,method_b=b,
                                     delta_a_minus_b_db=d.mean(),ci_low_db=np.quantile(boot,.025),ci_high_db=np.quantile(boot,.975)))
    pd.DataFrame(method_pairs).to_csv(out/'results/method_paired.csv',index=False)
    seed=new.groupby(['map_id','fraction','mechanism','method','seed'],as_index=False).hidden_rmse_db.mean()
    seed.groupby(['map_id','fraction','mechanism','method']).hidden_rmse_db.std().reset_index(name='seed_sd_db').to_csv(out/'results/seed_variability.csv',index=False)
    val=open_store(out,'budget_val_v02').export();curves=pd.read_csv(out/'results/budget_val_v02/training_curves.csv')
    grouped=curves.groupby(['map_id','method','fraction','mechanism','step'],as_index=False).agg(
                   observed_rmse_raw=('observed_rmse_raw','mean'),hidden_rmse_db=('hidden_rmse_db','mean'))
    grouped.to_csv(out/'results/validation_budget_region_curves.csv',index=False)
    for metric,title,filename in [('observed_rmse_raw','Full observed raw RMSE (dB)','budget_observed'),('hidden_rmse_db','Validation hidden RMSE (dB)','budget_hidden')]:
        fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
        for ri,mechanism in enumerate(['random','block']):
            for ci,fraction in enumerate([.01,.05,.1]):
                ax=axes[ri,ci]
                for method in ['lowrank','mlp']:
                    g=grouped[(grouped.method==method)&(grouped.fraction==fraction)&(grouped.mechanism==mechanism)&(grouped.step>=200)]
                    stats=g.groupby('step')[metric].agg(['mean','std'])
                    scale=100 if metric=='observed_rmse_raw' else 1
                    ax.plot(stats.index,scale*stats['mean'],label=LABELS[method])
                    ax.fill_between(stats.index,scale*(stats['mean']-stats['std']),scale*(stats['mean']+stats['std']),alpha=.1)
                ax.set_xscale('log');ax.set_title(f'{mechanism}, {fraction:.0%}');ax.grid(alpha=.25)
                if ri==1:ax.set_xlabel('Completed parameter updates (log)')
                if ci==0:ax.set_ylabel(title)
        axes[0,2].legend(fontsize=8);fig.suptitle('budget_val_v02: 200–3000 updates; fixed hyperparameters; shading = region SD')
        fig.tight_layout();save(fig,figs/filename)
    budget_region=val.groupby(['map_id','method','checkpoint_step'],as_index=False)[['observed_rmse_db','observed_rmse_raw','hidden_rmse_db','seconds','data_fit_mse','regularization','total_objective']].mean()
    budget_summary=budget_region.groupby(['method','checkpoint_step'],as_index=False).mean(numeric_only=True).drop(columns='map_id')
    budget_summary.to_csv(out/'results/budget_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for method in ['lowrank','mlp']:
        g=budget_summary[budget_summary.method==method]
        ax.plot(g.seconds,g.hidden_rmse_db,'o-',label=LABELS[method])
        for _,r in g.iterrows():ax.annotate(str(int(r.checkpoint_step)),(r.seconds,r.hidden_rmse_db),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Cumulative seconds including diagnostics');ax.set_ylabel('Validation hidden RMSE (dB)');ax.legend();ax.grid(alpha=.3)
    ax.set_title('budget_val_v02: accuracy / training cost');save(fig,figs/'budget_error_runtime')
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,mechanism in zip(axes,['random','block']):
        for method in METHODS:
            g=summaries['test_v02'];g=g[(g.method==method)&(g.mechanism==mechanism)].sort_values('fraction')
            ax.plot(100*g.fraction,g.rmse_db,'o-',label=LABELS[method]);ax.fill_between(100*g.fraction,g.ci_low_db,g.ci_high_db,alpha=.12)
        ax.set_xlabel('Observed outdoor pixels (%)');ax.set_title(mechanism);ax.grid(alpha=.3)
    axes[0].set_ylabel('Hidden RMSE (dB), region mean');axes[1].legend(fontsize=8)
    fig.suptitle('test_v02: exploratory follow-up on previously viewed holdout; 95% cluster bootstrap')
    fig.tight_layout();save(fig,figs/'performance_test_v02')
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,mechanism in zip(axes,['random','block']):
        for method in METHODS:
            g=paired[(paired.method==method)&(paired.mechanism==mechanism)].sort_values('fraction')
            ax.errorbar(100*g.fraction,g.delta_new_minus_baseline_db,
                        yerr=[g.delta_new_minus_baseline_db-g.ci_low_db,g.ci_high_db-g.delta_new_minus_baseline_db],fmt='o-',label=LABELS[method],capsize=3)
        ax.axhline(0,color='gray',lw=1);ax.set_title(mechanism);ax.set_xlabel('Observed outdoor pixels (%)');ax.grid(alpha=.3)
    axes[0].set_ylabel('New minus baseline RMSE (dB)');axes[1].legend(fontsize=8)
    fig.suptitle('Region-paired version differences; six previously viewed test regions')
    fig.tight_layout();save(fig,figs/'baseline_new_paired')
    # Spatial representative stays fixed to the already prespecified baseline case.
    old_case=json.loads((baseline/'results/case_selection.json').read_text())['representative']
    condition=(new.map_id==old_case['map_id'])&(new.tx_id==old_case['tx_id'])&(new.fraction==old_case['fraction'])&(new.mechanism==old_case['mechanism'])&(new.seed==old_case['seed'])
    records=new[condition].set_index('method');r=records.iloc[0];image=load_image(out,r)
    with np.load(out/r.mask_path) as f:obs=f['observed']
    valid=image['valid'];y=image['y'];fig,axes=plt.subplots(2,6,figsize=(19,7),constrained_layout=True)
    sig=axes[0,0].imshow(np.where(valid,-147+100*y,np.nan),vmin=-147,vmax=-47,cmap='viridis');axes[0,0].set_title('Saved truth')
    axes[0,1].imshow(np.where(valid,obs,np.nan),cmap='gray',vmin=0,vmax=1);axes[0,1].set_title('Observation mask')
    axes[1,0].imshow(image['buildings'],cmap='gray',vmin=0,vmax=255);axes[1,0].set_title('Buildings')
    axes[1,1].axis('off')
    for i,method in enumerate(METHODS):
        with np.load(out/records.loc[method,'prediction_path']) as f:p=np.clip(f['raw'],0,1)
        axes[0,i+2].imshow(np.where(valid,-147+100*p,np.nan),vmin=-147,vmax=-47,cmap='viridis');axes[0,i+2].set_title(LABELS[method])
        err=axes[1,i+2].imshow(np.where(valid&~obs,100*np.abs(p-y),np.nan),vmin=0,vmax=30,cmap='magma')
        axes[1,i+2].set_title(f"Hidden RMSE {records.loc[method,'hidden_rmse_db']:.2f} dB")
    for ax in axes.flat:
        if ax.axison:ax.set_xlabel('column (m)');ax.set_ylabel('row (downward, m)')
    fig.colorbar(sig,ax=axes[0,:],label='Saved pathgain (dB)',shrink=.6)
    fig.colorbar(err,ax=axes[1,2:],label='Absolute hidden error (dB), saturated at 30',shrink=.6)
    fig.suptitle(f"test_v02: map {old_case['map_id']}, Tx {old_case['tx_id']}; random 5%, seed 11; fixed baseline representative")
    fig.text(.5,-.015,'RadioMapSeer: Yapar, Levie, Kutyniok & Caire; https://radiomapseer.github.io/; https://creativecommons.org/licenses/by/4.0/.\nAdapted masking/colors/predictions/errors. Layouts: © OpenStreetMap contributors; https://www.openstreetmap.org/copyright.',ha='center',fontsize=8)
    save(fig,figs/'reconstruction_test_v02')
    write_report(out,baseline,summaries,paired,budget_summary)
    inputs={'new_results':file_sha(out/'results/test_v02/results.csv'),'baseline_results':file_sha(baseline/'results/main.csv'),
            'validation_curves':file_sha(out/'results/budget_val_v02/training_curves.csv')}
    outputs={str(p.relative_to(out)):file_sha(p) for folder in ['figures','reports'] for p in (out/folder).glob('*') if p.is_file() and p.suffix in ['.png','.svg']}
    atomic_json(out/'reports/presentation_manifest.json',dict(inputs=inputs,figure_sha256=outputs,
                summaries='Recomputed from version-bound records; report tables rendered from these summaries',completed_utc=utc()))
    print('Version-labelled summaries, paired bootstrap, diagnostics figures and updated report generated',flush=True)

def write_report(out,baseline,summaries,paired,budget):
    selection=json.loads((out/'configs/selected_v02.json').read_text());gate=json.loads((out/'reports/further_tuning_gate.json').read_text())
    candidates=pd.read_csv(out/'reports/budget_candidate_scores.csv')
    compact=candidates[['method','checkpoint_step','rmse_db']].copy()
    compact['reg']=[json.loads(x)['reg'] for x in candidates.params_json]
    new=summaries['test_v02'];old=summaries['baseline_v01']
    observed=budget[['method','checkpoint_step','observed_rmse_db','hidden_rmse_db','data_fit_mse','regularization','total_objective','seconds']]
    smooth_idw=new.pivot(index=['fraction','mechanism'],columns='method',values='rmse_db')
    diff=smooth_idw['smooth']-smooth_idw['idw']
    checkpoint_count=1296+ (432 if gate['triggered'] else 0)
    report=f'''# SparseRadioMap v0.1.0: Corrections and Training-Budget Diagnostics

Author: Yuntao Xue. Experimental versions: baseline_v01 → test_v02. This comparison is a **follow-up exploratory evaluation on the same previously viewed holdout**, not a new confirmatory test.

## 1. Preserved baseline and implementation fixes

The original experiment used 30 regions and 60 no-car DPM images, with an 18/6/6 development/validation/test region split. Its 16 pipeline checks, 216 tuning runs, 864 main reconstructions and 200-step neural training budget are preserved. Original result directories were not overwritten. The complete local `history/baseline_v01/snapshot.zip` preserves source, configurations, records, masks, predictions, reports and figures; `snapshot_manifest.json` records per-file SHA-256 hashes. The original ZIP was reused without downloading it again, and its SHA-256 was verified as `c80672cea3b3693d4a0a57eefb36a1a7e3ef78e04a99e6842d5a96afb6cc072f`.

The first-run fix explicitly creates `configs`, `reports` and `data/processed` before audit outputs are written, and removes a stale completion marker when an audit starts. The regression test calls `audit()` in a temporary directory with a small, valid 256×256 grayscale ZIP, no network access and no pre-existing processed directory. Environment-recording and download directories are also explicitly created.

The legacy unversioned runner is disabled to prevent new source-dependent run IDs from being mixed with old records. Expected-task manifests bind source, parameters, configuration, data and masks. Exports reject extra or missing records, source changes, identity mismatches and file-hash mismatches. Verified cache reuse and new fits have separate counters. A successful cache-only invocation is not evidence of independent reproduction.

## 2. Data, valid pixels and physical encoding

RadioMapSeer contains propagation simulations based on urban layouts, not field measurements. This project uses only DPM without cars at the native 256×256 resolution and 1 m/pixel. Outdoor valid pixels satisfy `building PNG == 0`; outdoor zero-gray pixels remain valid. The normalized target is `y=gray/255`, and saved negative pathgain is `-147+100*y` dB. Zero values are truncated and quantized, so pre-truncation truth cannot be recovered. The author's additional `thresh=0.2` transformation is not applied.

JSON `(x,y)` maps to PNG `(row=255-y, col=x)`. The complete original audit is retained in the local `reports/data_audit.md`. The new version independently regenerates processed arrays from the original ZIP without modifying raw data. The external filesystem does not enforce Unix read-only permissions: read-only use is a workflow rule backed by hash verification, not a claim that the volume is mounted read-only.

## 3. Prespecified budget experiment

Only development and validation data were used for selection; test rankings did not select hyperparameters. Development used the first two development regions in the original manifest, both Tx images per region, both mechanisms, all three fractions and seed 11: 48 training trajectories with 144 checkpoints. After estimating cost, validation used all six validation regions and 12 images, random sampling and 64×64 contiguous holes, observation fractions of 1%, 5% and 10%, and seeds 11/22/33. The two trainable methods produced 432 trajectories and 1,296 checkpoints.

Architecture, initialization, Fourier frequencies, rank, learning rate, regularization and random minibatch sequence were held fixed. Each trajectory saved predictions at 200, 1,000 and 3,000 updates. Prespecified logging steps were 0, 1, 10, 25, 50, 100, 200, 400, 600, 800, 1,000, 1,500, 2,000, 2,500 and 3,000. Each log records complete-observation MAE/RMSE before and after the common prediction clipping policy, data MSE, regularization, total objective, learning rate, elapsed and optimization time, raw prediction range and clipping fraction.

MLP minibatch loss is explicitly measured **before** the update; complete-observation diagnostics are measured **after** the update. Hidden truth is accessed only by an external scoring callback using detached NumPy predictions; callback results do not affect optimization. Legacy ordinary-reconstruction final-loss metadata comes from before the final update and is distinguished from post-update complete-observation diagnostics.

The low-rank objective is `MSE + reg*(mean(U²)+mean(V²))`. A fixed observed mean occupies one rank component; the remaining components represent the residual. MLP uses Adam with coupled L2 weight decay; its logged complete objective is `MSE + (reg/2)*sum(parameter²)`. A minibatch loss is not presented as that complete objective. Diagnostic timings include logging and scoring forward passes, while main reconstruction timings exclude scoring and file I/O; these timings are not interchangeable.

On real development images, the 200-step trajectory predictions exactly matched the original low-rank and MLP implementations, with maximum difference zero. Complete validation logs: [training_curves.csv](../results/budget_val_v02/training_curves.csv).

## 4. Observed budget results and configuration selection

The following values first average Tx, seeds, mechanisms and fractions equally within each region, then average across the six validation regions. Observed RMSE uses clipped predictions. Complete raw observation errors and objective decompositions remain available in the curve CSV.

{table(observed)}

![Complete-observation error](../figures/budget_observed.png)

![Validation hidden error](../figures/budget_hidden.png)

![Diagnostic error and runtime](../figures/budget_error_runtime.png)

Selection minimizes region-averaged hidden RMSE, balancing both mechanisms and all three fractions; ties prefer fewer steps. Settings were not selected separately for each test image. The frozen configuration is:

```json
{json.dumps(selection['methods'],indent=2)}
```

Freeze time: `{selection['frozen_utc']}`. Scientific source SHA-256: `{selection['scientific_source_sha256']}`. Full candidates, validation-plan hash, configuration provenance and selection rationale are recorded in [selected_v02.json](../configs/selected_v02.json).

The prespecified additional-tuning gate requires complete-observation raw low-rank RMSE to be at least 0.04 in normalized units at every budget. Gate triggered: **{gate['triggered']}**. Recorded values: `{gate['observed_rmse_by_step']}`. If triggered, only `reg=0.01` and `0.1` would be compared at the budget-selected step count, keeping rank, learning rate and architecture fixed and using the same complete validation conditions. The search was not expanded. Additional validation checkpoints: **{432 if gate['triggered'] else 0}**. The complete candidate table follows; candidates with different regularization must not be treated as checkpoints on one trajectory. The table rounds values to four decimals; exact regularization values are preserved in the configuration JSON and candidate CSV.

{table(compact)}

**Observations:** The tables and condition-specific curves show how complete observed and hidden errors change with budget. MLP observation fitting improves with more updates, but its aggregate validation hidden error increases. Low-rank hidden error improves slightly while observation error plateaus.

**Interpretation:** Falling observation error with rising hidden error suggests overfitting or extrapolation problems under the evaluated model and missingness mechanisms; it does not justify automatically choosing the longest budget. Improvements in both errors would support an insufficient original budget. A low-rank plateau may reflect regularization, optimization or rank limitations, but does not establish inadequate representation capacity. The low-rank regularization term is approximately 0.00438 and data term approximately 0.00176, consistent with a regularization tradeoff.

**Unverified hypotheses:** No rank/architecture ablation, learning-rate search or new model was executed, so those possible explanations are not reported as findings. The gate is a budget-management rule, not a physical-reliability or statistical-significance threshold. A false gate does not establish optimal regularization, and the causal role of regularization was not tested by ablation.

## 5. Holdout results after configuration freeze

The complete `test_v02` comparison contains 12 images × 3 fractions × 2 mechanisms × 3 seeds × 4 methods = **864 new fits**. No previous fitted outputs were reused. Low-rank hidden RMSE averaged equally across six conditions decreased from 11.044 to 10.820 dB, approximately 0.224 dB, without changing the aggregate method ranking under the evaluated budget. Paired improvements at 1%/5% are approximately 0.24–0.37 dB; at 10% they are only approximately 0.004–0.016 dB. Their practical magnitude should not be overstated.

MLP, IDW and smoothing configurations were unchanged, and their new predictions matched the baseline. Historical results were used only for subsequent paired comparisons and independent checks. Training does not read hidden truth. Raw predictions and shared masks are retained locally in separate version directories. Parameters were frozen before viewing this new holdout comparison, but the original holdout had already been viewed, so this remains a follow-up exploratory evaluation.

{table(new[['version','fraction','mechanism','method','mae_db','rmse_db','bias_db','observed_rmse_db','hole_rmse_db','gradient_rmse_db','seconds']])}

![New-version main results](../figures/performance_test_v02.png)

![Region-level paired version differences](../figures/baseline_new_paired.png)

Paired differences are new minus baseline; negative values favor the new version:

{table(paired)}

Validation subconditions show that MLP hidden RMSE under random 10% observations decreases from 4.675 to 4.358 dB, while under contiguous-hole 5% observations it increases from 4.988 to 5.354 dB; 1% conditions also worsen. Choosing 200 steps reflects the balanced aggregate criterion, not a claim that every condition favors shorter training.

Regions are the aggregation and bootstrap units. The 2,000 bootstrap resamples draw entire regions while retaining their Tx/seed structure. Intervals based on six holdout regions are exploratory; pixels are not sampled as independent observations. New-version smoothing minus IDW RMSE ranges from {diff.min():.4f} to {diff.max():.4f} dB, averaging {diff.mean():.4f} dB. These small effect sizes must be reported; an exploratory interval excluding zero alone does not establish a substantial practical advantage. High-gradient scoring retains the prespecified top 10% of outdoor finite-difference gradients. Hidden, observed and hole-interior locations are scored separately. CSVs also record unclipped RMSE and clipping fractions.

![New-version reconstruction](../figures/reconstruction_test_v02.png)

The spatial example retains the baseline's prespecified median representative, without selecting an attractive image from new results. Signal color limits are shared at −147 to −47 dB; error limits are 0 to 30 dB. Attribution and modifications are documented in [FIGURE_ATTRIBUTION.md](FIGURE_ATTRIBUTION.md).

## 6. Genuine first runs, independent reproduction and consistency

Fresh-run commands require an empty output directory and read only the hash-verified original ZIP to process a real image, generate masks, fit all four methods, score and export. Existing processed arrays, records and predictions are not fitting inputs. Reference records are accessed only after new fitting finishes. Actual records are preserved in `experiments/fresh_dev_v02/reports/fresh_verification.json` and `experiments/fresh_test_v02/reports/fresh_verification.json`. The development example reproduces the baseline 200-step configuration; the holdout example reproduces the final frozen configuration. Each executed four real fits with zero cache hits.

The minimal public staging source passed all 15 tests and executed four fresh fits in another empty directory. Its package contains no reference arrays; after fitting, outputs were separately compared against the complete local baseline. Masks, predictions and metrics matched exactly, as recorded in [public_candidate_fresh_verification.json](public_candidate_fresh_verification.json). An explicit `--cpu` fresh check also executed four fits and matched its baseline exactly. The revision executed **1,364 fitting calls**, excluding unit tests. Its 1,440 diagnostic checkpoints share 480 trajectories and are not independent fitting calls.

Using the complete local truth and raw outputs, all 864 baseline and all 864 new-version main metric records were recomputed independently and compared against CSV and JSON. Checks also covered 216 shared masks, equal observation counts and selected parameters. Evidence is in [score_verification.json](score_verification.json) and the `*_recomputed.csv` files. Figures and report tables derive from the version-specific summaries; `presentation_manifest.json` records input CSV and figure hashes.

`requirements-lock.txt` is a **pinned direct-dependency list**, not a complete transitive or wheel-hash lock. A clean Python 3.12 environment without system-site-packages was created, installed from the pins and used for tests and real fresh runs. Resolved versions are recorded in `clean_environment.json`. The local experimental environment is distinguished from user installation instructions; no `.venv` is supplied. **Only CPU was actually tested; cross-device determinism is unverified.**

## 7. Data licensing, packaging and citation

The earlier statement that no clear data license had been found missed a link in the official website HTML and is superseded. Next to the original download, `rel=license` points to [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), verifiable at line 36 of the [official source](https://raw.githubusercontent.com/radiomapseer/radiomapseer.github.io/master/index.html). Official HTML, repository source, author instructions and hashes are retained locally. Data licensing is not inferred from the paper license or RadioUNet's MIT code license. See [DATA_LICENSE.md](DATA_LICENSE.md).

The original ZIP is reused read-only locally and excluded from the public candidate. Processed arrays, masks and raw predictions are included in the complete local package and excluded from the public candidate. Numerical summaries and necessary verification tables are public. Selected dataset-derived display figures retain the official CC BY 4.0 notice, credit Çağkan Yapar, Ron Levie, Gitta Kutyniok and Giuseppe Caire, link the paper, and describe masking, color mapping, predictions and error-map modifications. Underlying layouts retain © OpenStreetMap contributors attribution and the upstream copyright/license link. No author endorsement is implied.

Original code remains MIT, separate from the data license. Detailed historical records remain in local `history`; public contents are compact, with reproducible configurations and no environments, credentials, caches, machine-specific absolute paths or large prediction arrays. The minimal candidate is published to GitHub under user authorization. No GitHub Release, PyPI package or website deployment was created, and the data authors were not contacted.

## 8. Limitations and future work

Evidence is limited to the evaluated two trainable representations and finite candidates, no-car DPM simulations, noiseless observations, six validation regions and six previously viewed holdout regions. It does not establish that low-rank models or neural fields are generally unsuitable for radio maps. No independent new confirmatory map set was evaluated, geographic overlap remains unchecked, pre-truncation truth is unavailable and GPU execution is untested. Building-boundary priors, censoring models, uncertainty and DeepMIMO are future work, not completed contributions.

## 9. Commands

From the project root:

```bash
python scripts/pipeline.py all --output experiments/v02 --archive data/raw/RadioMapSeer.zip
python scripts/pipeline.py fresh --output experiments/fresh_new --archive data/raw/RadioMapSeer.zip --fresh-split dev
```

Fresh output must be empty. Use `--fresh-split test` for an independent example under the final configuration. Stage commands are `prepare`, `dev-budget`, `val-budget`, `ridge`, `select`, `main`, `verify` and `summarize`. Version directories isolate outputs and cache verification counts. Scientific-source changes require new output directories and a new configuration freeze; old selections cannot silently be reused.

Software version: **v0.1.0**. `baseline_v01` and `test_v02` identify experiment versions. The 864 main fits are computational runs, not 864 independent statistical samples; the statistical unit is the map region.
'''
    (out/'reports/report.md').write_text(report)
