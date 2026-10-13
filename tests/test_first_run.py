"""Real ZIP-backed regression: no processed/configs directories pre-exist."""
from io import BytesIO
import hashlib,json,zipfile
import numpy as np
from PIL import Image
from sparseradiomap.data import audit,read_manifest,load_image

def test_audit_first_run_from_small_valid_zip(tmp_path):
    raw=tmp_path/'data/raw';raw.mkdir(parents=True)
    reports=tmp_path/'reports';reports.mkdir()
    archive=raw/'RadioMapSeer.zip'
    def encode(a):
        out=BytesIO();Image.fromarray(a).save(out,format='PNG');return out.getvalue()
    b=np.zeros((256,256),np.uint8);b[:20,:20]=255
    antenna=np.zeros_like(b);antenna[155,100]=255
    gain=np.broadcast_to(np.arange(256,dtype=np.uint8),(256,256)).copy()
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('gain/DPM/0_0.png',encode(gain))
        z.writestr('png/buildings_complete/0.png',encode(b))
        z.writestr('png/antennas/0_0.png',encode(antenna))
        z.writestr('antenna/0.json',json.dumps([[100,100]]))
    meta=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (reports/'download.json').write_text(json.dumps(meta))
    config=dict(selection_seed=19,map_counts={'dev':1,'val':0,'test':0},tx_per_map=1)
    assert not (tmp_path/'data/processed').exists()
    assert not (tmp_path/'configs').exists()
    audit(tmp_path,config)
    rows=read_manifest(tmp_path)
    assert len(rows)==1 and rows[0]['split']=='dev'
    image=load_image(tmp_path,rows[0])
    assert image['y'].shape==(256,256)
    assert not image['valid'][:20,:20].any()
    assert (tmp_path/'reports/audit_completed.json').exists()
    assert (tmp_path/'configs/split.json').exists()
