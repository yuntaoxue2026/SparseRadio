"""Immutable experiment plans and strictly checked, version-isolated records."""
from pathlib import Path
import datetime,hashlib,json,re,time,zipfile
import numpy as np
import pandas as pd
from .core import sampling,metrics,decode_gray
from .data import png,write_csv,load_image
from .methods import reconstruct

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(obj):return hashlib.sha256(canonical(obj).encode()).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def scientific_hash():
    # Includes sampling/scoring, processing, algorithms, training and selection.
    # Presentation/documentation are separately captured in release source snapshots.
    folder=Path(__file__).parent
    names=['core.py','data.py','methods.py','training.py','versioned.py','diagnostic.py']
    return digest({n:file_sha(folder/n) for n in names if (folder/n).exists()})
def array_sha(arrays):
    h=hashlib.sha256()
    for name,a in sorted(arrays.items()):
        a=np.ascontiguousarray(a);h.update(name.encode());h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=True));temp.replace(path)
def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.stem+'.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(path)

def prepare_dataset(output,archive,rows,expected_sha,verify_archive=True):
    """Read only ZIP; derive everything in output, never borrow processed files."""
    output=Path(output);archive=Path(archive)
    for d in ['data/processed','reports','configs']:(output/d).mkdir(parents=True,exist_ok=True)
    if verify_archive and file_sha(archive)!=expected_sha:raise ValueError('Raw archive SHA-256 mismatch')
    original=[]
    with zipfile.ZipFile(archive) as z:
        for row in rows:
            r=row.copy();blobs={k:z.read(row[k+'_path']) for k in ['gain','building','antenna']}
            if hashlib.sha256(blobs['gain']).hexdigest()!=row['sha256']:raise ValueError('Gain identity mismatch')
            gray,mode=png(blobs['gain']);b,bmode=png(blobs['building']);tx,tmode=png(blobs['antenna'])
            if gray.shape!=(256,256) or mode!=bmode or mode!=tmode or mode!='L':raise ValueError('Invalid images')
            arrays=dict(y=decode_gray(gray).astype(np.float32),valid=b==0,buildings=b,antenna=tx)
            r['logical_data_sha256']=array_sha(arrays)
            r['building_sha256']=hashlib.sha256(blobs['building']).hexdigest()
            r['antenna_sha256']=hashlib.sha256(blobs['antenna']).hexdigest()
            path=output/'data/processed'/f"{r['map_id']}_{r['tx_id']}.npz"
            if path.exists():
                with np.load(path) as f:
                    if array_sha({k:f[k] for k in f.files})!=r['logical_data_sha256']:raise ValueError('Processed data changed')
            else:atomic_npz(path,**arrays)
            original.append(r)
    write_csv(output/'data/manifest.csv',original)
    atomic_json(output/'reports/processing.json',dict(archive_sha256=expected_sha,images=len(original),
                input='Read-only original ZIP; no existing processed arrays used',completed_utc=utc(),
                dataset_sha256=digest([{k:r[k] for k in ['map_id','tx_id','split','sha256','logical_data_sha256']} for r in original])))
    return original

def task(root,config,row,method,params,fraction,mechanism,seed,**extra):
    image=load_image(root,row)
    mask_seed=int(np.random.SeedSequence([config['selection_seed'],row['map_id'],row['tx_id'],seed]).generate_state(1)[0])
    obs,hole,bounds=sampling(image['valid'],fraction,mechanism,mask_seed,config['hole_size'])
    mask_sha=array_sha({'observed':obs,'hole':hole})
    return dict(map_id=row['map_id'],tx_id=row['tx_id'],split=row['split'],method=method,
                params=params,fraction=fraction,mechanism=mechanism,seed=seed,mask_seed=mask_seed,
                hole_size=config['hole_size'],mask_sha256=mask_sha,data_sha256=row['sha256'],
                logical_data_sha256=row['logical_data_sha256'],**extra)

class Store:
    def __init__(self,root,version,config,tasks):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',version):raise ValueError('Invalid version identifier')
        self.root=Path(root);self.version=version;self.config=config
        self.directory=self.root/'results'/version
        self.code=scientific_hash()
        self.plan=dict(version=version,code_sha256=self.code,config_sha256=digest(config),tasks=tasks)
        self.plan_sha=digest(self.plan)
        for d in ['runs','predictions','masks','trajectories']:(self.directory/d).mkdir(parents=True,exist_ok=True)
        path=self.directory/'run_manifest.json'
        if path.exists():
            old=json.loads(path.read_text())
            if old['plan']!=self.plan or old['plan_sha256']!=self.plan_sha:raise ValueError('Version already frozen with different source/config/data/tasks; use a new version')
        else:atomic_json(path,dict(plan=self.plan,plan_sha256=self.plan_sha,frozen_utc=utc()))
        self.entries={digest(dict(plan_sha256=self.plan_sha,task=t)):t for t in tasks}
        if len(self.entries)!=len(tasks):raise ValueError('Duplicate tasks in run manifest')
        self.ensure_no_extra()
    def ensure_no_extra(self):
        actual={p.stem for p in (self.directory/'runs').glob('*.json')}
        if actual-set(self.entries):raise ValueError('Unexpected/stale run records in version directory')
    def check_source(self):
        if scientific_hash()!=self.code:raise ValueError('Scientific source changed after version freeze')
    def masks(self,t):
        image=load_image(self.root,t)
        if array_sha(image)!=t['logical_data_sha256']:raise ValueError('Data mismatch')
        obs,hole,bounds=sampling(image['valid'],t['fraction'],t['mechanism'],t['mask_seed'],t['hole_size'])
        if array_sha({'observed':obs,'hole':hole})!=t['mask_sha256']:raise ValueError('Mask definition mismatch')
        return image,obs,hole,bounds
    def cached(self,ident):
        path=self.directory/'runs'/f'{ident}.json'
        if not path.exists():return None
        rec=json.loads(path.read_text())
        if rec.get('status')=='failure':return None
        expected=self.entries[ident]
        if rec.get('identity')!=expected or rec.get('plan_sha256')!=self.plan_sha or rec.get('run_id')!=ident:raise ValueError('Identity/config/source mismatch')
        for field,sha in [('prediction_path','prediction_sha256'),('mask_path','mask_file_sha256')]:
            p=self.root/rec[field]
            if not p.exists() or file_sha(p)!=rec[sha]:raise ValueError('Missing or changed artifact: '+field)
        self.masks(expected)
        return rec
    def save(self,ident,raw,seconds,metadata=None,provenance='refitted'):
        t=self.entries[ident];image,obs,hole,bounds=self.masks(t)
        scored=metrics(raw,image['y'],image['valid'],obs,hole)
        prediction=self.directory/'predictions'/f'{ident}.npz'
        mask=self.directory/'masks'/f"{t['mask_sha256']}.npz"
        atomic_npz(prediction,raw=np.asarray(raw,dtype=np.float32))
        if mask.exists():
            with np.load(mask) as f:
                if array_sha({'observed':f['observed'],'hole':f['hole']})!=t['mask_sha256']:raise ValueError('Stored mask changed')
        else:atomic_npz(mask,observed=obs,hole=hole,bounds=np.array(bounds if bounds else [],dtype=int))
        rec=dict(t,identity=t,version=self.version,plan_sha256=self.plan_sha,run_id=ident,status='success',
                 seconds=seconds,utc=utc(),provenance=provenance,params_json=canonical(t['params']),
                 prediction_path=str(prediction.relative_to(self.root)),prediction_sha256=file_sha(prediction),
                 mask_path=str(mask.relative_to(self.root)),mask_file_sha256=file_sha(mask),**scored)
        rec.update(metadata or {});atomic_json(self.directory/'runs'/f'{ident}.json',rec)
        return rec
    def export(self,complete=True):
        self.check_source();self.ensure_no_extra();rows=[]
        for ident in self.entries:
            rec=self.cached(ident)
            if rec is None:
                if complete:raise ValueError('Missing/failed expected record: '+ident)
                continue
            rows.append({k:v for k,v in rec.items() if k not in ['identity','params']})
        write_csv(self.directory/'results.csv',rows)
        return pd.DataFrame(rows)
    def run(self):
        self.check_source();counts={'refitted':0,'cached_verified':0,'failed':0}
        for i,(ident,t) in enumerate(self.entries.items()):
            if self.cached(ident) is not None:counts['cached_verified']+=1
            else:
                image,obs,hole,_=self.masks(t);start=time.perf_counter()
                raw,meta=reconstruct(t['method'],image['y'].shape,np.argwhere(obs),image['y'][obs],t['params'],t['mask_seed'])
                self.save(ident,raw,time.perf_counter()-start,meta);counts['refitted']+=1
            if (i+1)%24==0:print(self.version,i+1,'/',len(self.entries),counts,flush=True)
        self.export();atomic_json(self.directory/'invocation.json',dict(counts,completed_utc=utc(),
                        note='A cache-only invocation is NOT an independent reproduction'))
        return counts

def open_store(root,version):
    data=json.loads((Path(root)/'results'/version/'run_manifest.json').read_text())
    plan=data['plan'];config=json.loads((Path(root)/'configs'/f'{version}.json').read_text())
    return Store(root,version,config,plan['tasks'])
