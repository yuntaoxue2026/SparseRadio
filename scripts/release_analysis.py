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
    report=f'''# SparseRadioMap v0.1 发布前修正与预算诊断

作者：Yuntao Xue。版本：baseline_v01 → test_v02。当前比较是**同一已查看保留集上的后续探索性评价**，不是全新确认性测试。

## 1. 保留的基线与修复

原始30区域、60无车DPM图，18/6/6开发/验证/测试划分，16流程检查、216调参、864主实验及200步预算均保存。原结果目录未覆盖；history/baseline_v01/snapshot.zip包含旧源码、配置、记录、掩码、预测、报告及图表，snapshot_manifest.json为逐文件SHA-256。原始ZIP再验证SHA-256为c80672cea3b3693d4a0a57eefb36a1a7e3ef78e04a99e6842d5a96afb6cc072f，无重复下载。

修复audit首次运行前显式创建configs/reports/data/processed，开始审计时移除过期完成标记。真实回归测试以临时目录、小型256×256灰度ZIP执行audit，无网络、无预建processed。环境记录及下载目录也显式创建。旧无版本runner拒绝运行，避免源码变化后在旧目录混入新ID。新版使用预期任务manifest：精确绑定源码、参数、配置、数据、掩码；多余、缺失、源变更、身份或文件哈希不符均拒绝汇总。缓存核验与重新拟合分别计数，缓存成功退出不算独立复现。

## 2. 数据、有效区与物理编码

RadioMapSeer是基于城市布局的传播仿真，不是真实测量。本次仅无车DPM；保持原生256×256、1m/pixel，室外有效区由建筑PNG==0定义，室外零灰度保留。y=gray/255，保存负pathgain=-147+100y dB；零值有截断及量化，不能恢复截断前真值。作者额外thresh=0.2未使用。JSON(x,y)映射(row=255-y,col=x)，全量原审计见本地reports/data_audit.md。新版直接从原始ZIP重新生成独立处理数组；没有修改raw。外置文件系统不强制Unix只读权限，逻辑只读加哈希核验，不声称只读挂载。

## 3. 预算实验预先定义

仅开发/验证，绝不据测试排名挑超参数。开发取原manifest前2开发区域、全部2Tx、两机制、三比例、种子11，48训练轨迹/144检查点。测得成本后使用全部6验证区域/12图、随机与64×64空白区、1/5/10%、种子11/22/33；两神经方法共432轨迹/1296检查点。

保持旧结构、初始化、Fourier频率、秩、学习率、正则及随机批次序列；一条轨迹保存200/1000/3000步。记录点预设为0、1、10、25、50、100、200、400、600、800、1000、1500、2000、2500、3000。每点记录完整观测MAE/RMSE（raw及统一裁剪后）、数据MSE、正则、总目标、lr、累计耗时、优化耗时、原始范围及裁剪比例。MLP小批量loss明确是更新前数值，完整观测指标是更新后；不能混淆。隐藏真值仅由 detached numpy 预测的外部评分回调读取，回调返回不影响训练。普通重建的旧final-loss元数据来自最后一次参数更新前，与诊断的更新后完整观测指标明确区别。

低秩目标MSE+reg·(mean(U²)+mean(V²))，固定观测均值占一个秩，其余秩为残差。MLP使用Adam的耦合L2 weight_decay；记录的完整目标为MSE+(reg/2)·sum(parameter²)，不是把小批量loss叫完整目标。诊断耗时包含日志/评分forward，主实验耗时不含评分/文件I/O，二者不可直接等同。

真实开发图200步轨迹与原实现预测最大差异为0（lowrank和MLP）。完整曲线：[training_curves.csv](../results/budget_val_v02/training_curves.csv)。

## 4. 实际预算结果与选择

以下先对每个区域内Tx、seed、两机制、三比例等权平均，再跨6验证区域平均；观测RMSE为裁剪后值，完整raw观测误差和目标分解可见曲线CSV。

{table(observed)}

![完整观测误差](../figures/budget_observed.png)

![验证隐藏误差](../figures/budget_hidden.png)

![诊断误差与耗时](../figures/budget_error_runtime.png)

选择标准为区域平均隐藏RMSE，平衡所有机制和比例；相同时优先较少步数。没有按测试图逐图择优。冻结配置为：

```json
{json.dumps(selection['methods'],indent=2)}
```

冻结时间：{selection['frozen_utc']}。科学实现SHA-256：{selection['scientific_source_sha256']}；完整候选、验证版本plan哈希、配置来源和选择依据见configs/selected_v02.json。

预设额外调参门槛：低秩所有预算的完整观测raw RMSE均≥0.04才检查有限正则候选；本次触发={gate['triggered']}，记录值={gate['observed_rmse_by_step']}。如触发，只在预算研究选定步数下比较reg=0.01和0.1，秩/学习率/架构不变，使用全部相同验证条件；没有扩大搜索。附加验证检查点为{432 if gate['triggered'] else 0}，完整候选表如下（不同正则不可当作同一预算轨迹）：

{table(compact)}

**观察与解释分开**：观察到的完整观测及隐藏误差趋势以上述表和各条件曲线为准。观测误差下降而隐藏误差上升提示当前模型/缺失机制下过拟合或外推问题，不能自动选最长训练；两者下降才支持原预算不足。低秩平台可能来自正则权衡、优化或秩限制，不等同已证明表示能力不足。没有运行秩/结构消融、学习率搜索或新的模型，不把这些假设写成结果。低秩正则项约0.00438、数据项约0.00176，较早进入平台；这与正则权衡一致，但未以消融验证因果。该筛查门槛是预算管理规则，不是物理可靠性或统计显著性阈值，未触发不等于已证明正则最优。

## 5. 冻结后的保留集结果

新版本test_v02完整12图×3比例×2机制×3seed×4方法=864次，全部重新拟合；未复用旧拟合结果。低秩在六条件等权平均的隐藏RMSE由11.044降至10.820dB，改善约0.224dB；仍未改变本预算下的整体方法排序。1%/5%的配对改善约0.24–0.37dB，10%仅约0.004–0.016dB，不宜夸大实际价值。MLP、IDW、平滑配置未变，重新拟合结果与基线相同。老版本仅用于事后配对比较与独立检查。训练不读取隐藏真值，全部原始预测和共享掩码保存在独立版本目录。参数选择已在查看本次保留集结果前冻结；但旧测试已看过，所以只能称后续探索性评价。

{table(new[['version','fraction','mechanism','method','mae_db','rmse_db','bias_db','observed_rmse_db','hole_rmse_db','gradient_rmse_db','seconds']])}

![新版主结果](../figures/performance_test_v02.png)

![区域级版本配对差异](../figures/baseline_new_paired.png)

配对差异new−baseline，负值表示新版较低：

{table(paired)}

验证子条件还显示：MLP随机10%下隐藏RMSE从4.675降至4.358dB，但区域5%从4.988升至5.354dB，1%也变差。总体选择200步体现平衡目标，不意味着所有条件都偏好短训练。

区域是汇总和bootstrap单位，2000次重抽整个区域，保留其Tx/seed结构。6区域区间是探索性；不按像素独立抽样。新版平滑−IDW差值范围{diff.min():.4f}至{diff.max():.4f}dB，均值{diff.mean():.4f}dB；应报告这种较小差异，不能因区间不跨零就称明显实际优势。高梯度评分继续采用预设室外差分模板前10%，隐藏、观测和空白区分开计分。CSV也记录裁剪前RMSE与裁剪比例。

![新版重建](../figures/reconstruction_test_v02.png)

空间展示沿用基线预设中位图，不按新结果挑好看图；统一信号色标−147至−47dB，误差0至30dB。数据署名及修改说明见FIGURE_ATTRIBUTION.md。

## 6. 真正首次运行、独立复现与一致性

首次运行命令要求空输出目录，只允许读取哈希核验后的原始ZIP，重新处理图、生成掩码、拟合四种方法、评分和导出；不读取已有processed/运行/预测作为输入。参考记录只在新拟合完成后用于比较。实际记录在experiments/fresh_dev_v02/reports/fresh_verification.json和experiments/fresh_test_v02/reports/fresh_verification.json。前者兼容基线200步，后者复现本次冻结配置。每次4个真实图条件拟合，cache=0。精简公开候选的实际源码也通过15项测试并在另一个空目录执行4次真实fresh拟合；公开包没有参考数组，拟合后另行与完整本地基线对照，掩码/预测/指标最大差异均为0，见public_candidate_fresh_verification.json。另实际执行显式--cpu的fresh检查4次，与对应基线完全一致。新增实际拟合调用总计1364（不计单元测试），诊断保存的1440个检查点共享480条轨迹，不当作独立拟合计数。

另外以已有完整真值重新计算baseline_v01和test_v02全部864条主指标，分别对照CSV与JSON，检查共享216掩码、观测数量和选定参数，见reports/score_verification.json及*_recomputed.csv。图和报告均由这些版本汇总表生成，presentation_manifest.json保存输入CSV和图表哈希。

requirements-lock.txt准确名称是**直接依赖版本固定清单**，不是含传递依赖/轮子哈希的完整锁。实际创建不继承system-site-packages的干净Python3.12环境，安装固定清单并运行测试/真实fresh任务；干净环境解析版本保存clean_environment.json。本地原实验环境与用户安装步骤分别说明，不宣称交付包含.venv。仅CPU实测，不推断跨设备确定性。

## 7. 数据许可、打包与引用

上一版“未找到明确许可”遗漏了官网HTML中的链接，现已修正。官方原始数据下载链接旁有`rel=license`指向[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，[官方源码](https://raw.githubusercontent.com/radiomapseer/radiomapseer.github.io/master/index.html)第36行可核实。本地保存官方HTML、仓库源码、作者说明和哈希；不从论文许可或RadioUNet MIT推断数据许可。见DATA_LICENSE.md。

原始数据：本地ZIP只读复用，不放公开候选。处理数组、掩码与原始预测：完整本地包包含，公开候选排除。数值汇总和必要验证表：公开候选包含。数据衍生展示图：按官网CC BY 4.0标记保留，说明mask/色标/预测/误差等改动、署名Yapar/Levie/Kutyniok/Caire并链接论文；不声称作者背书。原创代码仍MIT，许可证分开处理。详细旧历史在本地history，公开包只有精简展示与可复现配置，排除虚拟环境、凭据、缓存、本机绝对路径和大预测，没有自动上传或发布，也未联系作者。

## 8. 限制与未来工作

只评价当前两种神经表示及有限候选、无车DPM、无噪声观测、6个验证及6个已查看保留区域，不推出“低秩/神经场不适合无线地图”。没有独立的新确认性地图组，未核实地理重叠；截断前真值不可恢复，GPU未实测。后续可研究建筑边界先验、删失模型、不确定性和DeepMIMO；均未作为已完成贡献。

## 9. 命令

从项目根目录：`python scripts/pipeline.py all --output experiments/v02 --archive data/raw/RadioMapSeer.zip`。只会操作独立版本目录，缓存核验计数明确。真正独立从头运行：`python scripts/pipeline.py fresh --output experiments/fresh_new --archive data/raw/RadioMapSeer.zip --fresh-split dev`（必须为空）；最终配置独立复现用`--fresh-split test`。阶段命令：prepare、dev-budget、val-budget、ridge、select、main、verify、summarize。改科学源码后必须新输出/新冻结，不能沿用旧选择。
'''
    (out/'reports/report_zh.md').write_text(report)
