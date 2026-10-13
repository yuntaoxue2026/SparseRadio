import json
from pathlib import Path
import numpy as np
import pytest
from sparseradiomap.versioned import Store,task,array_sha
from sparseradiomap.core import sampling
from sparseradiomap.training import train_trajectory
from sparseradiomap.methods import reconstruct

@pytest.fixture
def store(tmp_path):
    d=tmp_path/'data/processed';d.mkdir(parents=True)
    y=np.linspace(.2,.8,256,dtype=np.float32).reshape(16,16)
    image=dict(y=y,valid=np.ones_like(y,dtype=bool),buildings=np.zeros_like(y,dtype=np.uint8),antenna=np.zeros_like(y,dtype=np.uint8))
    np.savez_compressed(d/'0_0.npz',**image)
    row=dict(map_id=0,tx_id=0,split='dev',sha256='fixture',logical_data_sha256=array_sha(image))
    cfg=dict(selection_seed=3,hole_size=4)
    t=task(tmp_path,cfg,row,'idw',{'k':8},.1,'random',11)
    return Store(tmp_path,'regression',cfg,[t])

def test_version_refuses_duplicate_and_unplanned_record(store):
    counts=store.run();assert counts['refitted']==1 and counts['cached_verified']==0
    counts=store.run();assert counts['refitted']==0 and counts['cached_verified']==1
    src=next((store.directory/'runs').glob('*.json'))
    (store.directory/'runs'/'unexpected.json').write_text(src.read_text())
    with pytest.raises(ValueError,match='Unexpected/stale'):store.export()

def test_version_refuses_missing_changed_artifact_or_identity(store):
    store.run();ident=next(iter(store.entries));p=store.directory/'runs'/f'{ident}.json'
    rec=json.loads(p.read_text());pred=store.root/rec['prediction_path']
    saved=pred.read_bytes();pred.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed artifact'):store.export()
    pred.write_bytes(saved);rec['identity']['params']['k']=1;p.write_text(json.dumps(rec))
    with pytest.raises(ValueError,match='mismatch'):store.export()

def test_version_refuses_source_change(store,monkeypatch):
    monkeypatch.setattr('sparseradiomap.versioned.scientific_hash',lambda:'changed')
    with pytest.raises(ValueError,match='Source changed|source changed'):store.run()
    with pytest.raises(ValueError,match='different source'):Store(store.root,store.version,store.config,list(store.entries.values()))

@pytest.mark.parametrize('method,cfg',[
    ('lowrank',dict(rank=4,lr=.03,reg=.05,steps=200,device='cpu')),
    ('mlp',dict(bands=3,width=32,lr=.003,reg=1e-5,batch_size=16,steps=200,device='cpu'))])
def test_checkpoint_same_trajectory_and_full_observed_telemetry(method,cfg):
    valid=np.ones((16,16),bool);obs,_,_=sampling(valid,.1,'random',12,4)
    rc=np.argwhere(obs);values=np.linspace(.2,.8,len(rc),dtype=np.float32)
    predictions,logs=train_trajectory(method,valid.shape,rc,values,cfg,17,[20,200],log_steps=[0,20,200])
    reference,_=reconstruct(method,valid.shape,rc,values,cfg,17)
    np.testing.assert_allclose(predictions[200],reference,atol=1e-7)
    for step in [20,200]:
        log=next(r for r in logs if r['step']==step)
        err=predictions[step][obs]-values
        assert log['observed_rmse_raw']==pytest.approx(np.sqrt(np.mean(err*err)),abs=1e-7)
        assert log['observed_mae_raw']==pytest.approx(np.mean(np.abs(err)),abs=1e-7)
        assert log['metrics_timing']=='after_parameter_update'
        assert log['total_objective']==pytest.approx(log['data_fit_mse']+log['regularization'])
    # Callback return cannot affect fitting, even when it claims hidden labels.
    other,_=train_trajectory(method,valid.shape,rc,values,cfg,17,[200],lambda *a:np.full(valid.shape,99),log_steps=[200])
    np.testing.assert_array_equal(other[200],predictions[200])
