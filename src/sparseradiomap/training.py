"""One seeded trajectory, post-update full-observation diagnostics, no truth input."""
import time
import numpy as np
import torch
from torch import nn
from .methods import positional,device_name

def train_trajectory(method,shape,rc,values,cfg,seed,checkpoints=(200,1000,3000),
                     callback=None,log_steps=None):
    """callback(step, raw, diagnostics) is scoring-only; its return is ignored.

    Model/optimizer have no references to hidden truth. Scoring callback receives
    detached numpy predictions only; it cannot change the optimizer or budget.
    Timing includes diagnostic forward passes, separately reports optimization.
    """
    torch.set_num_threads(1)
    dev=device_name() if cfg.get('device','auto')=='auto' else cfg['device']
    if dev!='cpu':
        raise RuntimeError('This validated trajectory implementation is CPU-only; GPU has not been tested')
    torch.manual_seed(seed)
    values=np.asarray(values,dtype=np.float32)
    rc=np.asarray(rc,dtype=np.int64)
    y=torch.as_tensor(values,device=dev)
    lr=cfg['lr']; reg=cfg['reg']
    start=time.perf_counter()
    if method=='lowrank':
        h,w=shape;rank=cfg['rank']
        if rank<2:raise ValueError('Rank includes fixed mean component')
        mean=float(values.mean())
        u=nn.Parameter(.03*torch.randn(h,rank-1,device=dev))
        v=nn.Parameter(.03*torch.randn(w,rank-1,device=dev))
        inds=torch.as_tensor(rc,dtype=torch.long,device=dev)
        params=[u,v]
        opt=torch.optim.Adam(params,lr=lr)
        def fit():return mean+(u[inds[:,0]]*v[inds[:,1]]).sum(1)
        def penalty():return reg*(u.square().mean()+v.square().mean())
        def predict():return (mean+u@v.T).cpu().numpy()
        rng=None
    elif method=='mlp':
        scale=torch.tensor(np.array(shape)-1,dtype=torch.float32,device=dev)
        x=positional(torch.as_tensor(rc,dtype=torch.float32,device=dev)/scale*2-1,cfg['bands'])
        width=cfg['width']
        model=nn.Sequential(nn.Linear(2+4*cfg['bands'],width),nn.ReLU(),
                            nn.Linear(width,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),
                            nn.Linear(width,1)).to(dev)
        params=list(model.parameters())
        opt=torch.optim.Adam(params,lr=lr,weight_decay=reg)
        def fit():return model(x).squeeze(1)
        def penalty():return .5*reg*sum(p.square().sum() for p in params)
        target=np.indices(shape).reshape(2,-1).T
        features=[positional(torch.as_tensor(c,dtype=torch.float32,device=dev)/scale*2-1,cfg['bands']) for c in np.array_split(target,max(1,len(target)//4096))]
        def predict():return np.concatenate([model(c).squeeze(1).cpu().numpy() for c in features]).reshape(shape)
        rng=np.random.default_rng(seed)
    else:raise ValueError(method)
    maxstep=max(checkpoints)
    logs=set(log_steps if log_steps is not None else [0,1,10,25,50,100,200,400,600,800,1000,1500,2000,2500,3000])
    logs.update(checkpoints);logs={s for s in logs if s<=maxstep}
    optimization_seconds=0.0;history=[];predictions={}
    minibatch=np.nan
    def record(step):
        with torch.no_grad():
            fitted=fit();err=fitted-y
            data=float(err.square().mean().cpu())
            regularization=float(penalty().cpu())
            raw=predict().copy()
            clipped=torch.clamp(fitted,0,1)-y
        info=dict(step=step,metrics_timing='after_parameter_update' if step else 'initialization',
                  observed_mae_raw=float(err.abs().mean().cpu()),observed_rmse_raw=float(np.sqrt(data)),
                  observed_mae_clipped=float(clipped.abs().mean().cpu()),
                  observed_rmse_clipped=float(clipped.square().mean().sqrt().cpu()),
                  data_fit_mse=data,regularization=regularization,total_objective=data+regularization,
                  minibatch_mse_before_update=float(minibatch),lr=float(opt.param_groups[0]['lr']),
                  elapsed_seconds=time.perf_counter()-start,optimization_seconds=optimization_seconds,
                  raw_min=float(raw.min()),raw_max=float(raw.max()),
                  clip_fraction_all_pixels=float(((raw<0)|(raw>1)).mean()),device=dev)
        history.append(info)
        if step in checkpoints:predictions[step]=raw
        if callback is not None:callback(step,raw,info.copy())
    if 0 in logs:record(0)
    for step in range(1,maxstep+1):
        t=time.perf_counter();opt.zero_grad(set_to_none=True)
        if method=='lowrank':
            data=(fit()-y).square().mean();loss=data+penalty()
        else:
            batch=torch.as_tensor(rng.choice(len(y),min(cfg['batch_size'],len(y)),replace=False),device=dev)
            data=(model(x[batch]).squeeze(1)-y[batch]).square().mean();loss=data
        minibatch=float(data.detach().cpu())
        loss.backward();opt.step();optimization_seconds+=time.perf_counter()-t
        if step in logs:record(step)
    return predictions,history
