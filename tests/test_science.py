import inspect
import numpy as np
import pytest
from sparseradiomap.core import decode_gray,to_db,sampling,metrics
from sparseradiomap.methods import reconstruct

def test_saved_units_and_floor():
    x=decode_gray(np.array([0,51,255],dtype=np.uint8))
    np.testing.assert_allclose(to_db(x),[-147,-127,-47])
    assert x[0]==0  # saved floor only; no pre-truncation inverse

def test_paired_masks_and_reproducibility():
    valid=np.ones((32,32),bool);valid[4:8,4:8]=False
    a,_,_=sampling(valid,0.1,'random',82,8)
    b,hole,bounds=sampling(valid,0.1,'block',82,8)
    b2,hole2,bounds2=sampling(valid,0.1,'block',82,8)
    assert a.sum()==b.sum()==int(valid.sum()*0.1)
    assert not (b&hole).any() and not (b&~valid).any()
    assert not (b&(valid&~b)).any()
    assert np.array_equal(b,b2) and np.array_equal(hole,hole2) and bounds==bounds2
    assert hole.sum()==64

def test_hidden_scoring_excludes_provided_points():
    y=np.full((8,8),0.2); raw=y.copy();obs=np.zeros_like(y,dtype=bool)
    obs[0]=True;raw[0]=0.8
    r=metrics(raw,y,np.ones_like(obs),obs,np.zeros_like(obs))
    assert r['hidden_rmse']==0 and r['observed_rmse']==pytest.approx(0.6)

def test_fit_api_and_hidden_truth_invariance():
    assert list(inspect.signature(reconstruct).parameters)==['method','shape','rc','values','cfg','seed']
    y=np.arange(64).reshape(8,8)/64;rc=np.array([[0,0],[2,2],[7,7]])
    observed=y[rc[:,0],rc[:,1]].copy()
    p,_=reconstruct('idw',y.shape,rc,observed,{'k':3},10)
    y[:]=999
    q,_=reconstruct('idw',y.shape,rc,observed,{'k':3},10)
    np.testing.assert_array_equal(p,q)

@pytest.mark.parametrize('method,cfg',[('idw',{'k':8}),('smooth',{'lam':0.1}),
    ('lowrank',{'rank':2,'steps':150,'device':'cpu'}),
    ('mlp',{'bands':2,'width':32,'steps':150,'device':'cpu'})])
def test_constant_known_signal(method,cfg):
    valid=np.ones((16,16),bool)
    obs,_,_=sampling(valid,0.2,'random',123,4)
    pred,_=reconstruct(method,valid.shape,np.argwhere(obs),np.full(obs.sum(),0.4),cfg,12)
    assert np.isfinite(pred).all()
    assert np.sqrt(np.mean((pred-0.4)**2))<0.04

def test_actual_split_no_region_leakage():
    from pathlib import Path
    from sparseradiomap.data import read_manifest
    root=Path(__file__).resolve().parents[1]
    if not (root/'data/manifest.csv').exists():
        pytest.skip('Dataset has not been downloaded/audited')
    rows=read_manifest(root)
    regions={s:{r['map_id'] for r in rows if r['split']==s} for s in ['dev','val','test']}
    assert list(map(len,regions.values()))==[18,6,6]
    assert not regions['dev']&regions['val'] and not regions['dev']&regions['test'] and not regions['val']&regions['test']
    assert len(rows)==60
