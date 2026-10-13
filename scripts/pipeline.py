"""Version-isolated experiments and a genuine ZIP-only fresh-run entry point."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR','/tmp/sparseradiomap-mpl-v02')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(ROOT/'src'))
from sparseradiomap.versioned import prepare_dataset,atomic_json,Store,task,open_store,file_sha,utc
from sparseradiomap.data import read_manifest
from sparseradiomap.diagnostic import planned,trajectories,regularization_plan,choose,main_store
EXPECTED_ZIP='c80672cea3b3693d4a0a57eefb36a1a7e3ef78e04a99e6842d5a96afb6cc072f'

def initialize(out,archive,force_cpu=False):
    if not archive.exists():
        from download import main as download
        download()
    rows=read_manifest(ROOT)
    prepare_dataset(out,archive,rows,EXPECTED_ZIP)
    for src,dest in [('experiment.json','protocol.json'),('selected.json','baseline_selected.json')]:
        data=json.loads((ROOT/'configs'/src).read_text());path=out/'configs'/dest
        if force_cpu and dest=='baseline_selected.json':
            for method in ['lowrank','mlp']:data['methods'][method]['device']='cpu'
        if path.exists() and json.loads(path.read_text())!=data:raise ValueError('Output protocol differs; choose a new output directory')
        if not path.exists():atomic_json(path,data)
    return json.loads((out/'configs/protocol.json').read_text()),json.loads((out/'configs/baseline_selected.json').read_text())

def budget(out,c,b,split):
    version='budget_dev_v02' if split=='dev' else 'budget_val_v02'
    if (out/'results'/version/'run_manifest.json').exists():s=open_store(out,version)
    else:s=planned(out,c,b,split,version)
    trajectories(s);return s

def ridge(out,c,b,val):
    version='ridge_val_v02'
    if (out/'results'/version/'run_manifest.json').exists():s=open_store(out,version)
    else:s=regularization_plan(out,c,b,val,version)
    if s is not None:trajectories(s)
    return s

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['all','prepare','audit','dev-budget','val-budget','ridge','select','main','summarize','verify','fresh'])
    p.add_argument('--output',type=Path,default=ROOT/'experiments/v02')
    p.add_argument('--archive',type=Path,default=ROOT/'data/raw/RadioMapSeer.zip')
    p.add_argument('--fresh-split',choices=['dev','test'],default='dev')
    p.add_argument('--cpu',action='store_true',help='Explicit CPU parameters for a NEW output/fresh run; use on GPU-capable hosts. Never changes an existing freeze.')
    args=p.parse_args();out=args.output.resolve();archive=args.archive.resolve()
    if out==ROOT:raise ValueError('Refuse to overwrite baseline root; use a separate output directory')
    if args.command=='fresh':
        from verify_release import fresh
        fresh(out,archive,args.fresh_split,ROOT,force_cpu=args.cpu);return
    if args.command in ['all','prepare','audit']:c,b=initialize(out,archive,args.cpu)
    else:
        c=json.loads((out/'configs/protocol.json').read_text());b=json.loads((out/'configs/baseline_selected.json').read_text())
        if args.cpu and any(b['methods'][m].get('device')!='cpu' for m in ['lowrank','mlp']):
            raise ValueError('--cpu requires a new output prepared with --cpu; cannot amend an existing frozen configuration')
    if args.command in ['all','dev-budget']:budget(out,c,b,'dev')
    if args.command in ['all','val-budget']:budget(out,c,b,'val')
    if args.command in ['all','ridge','select']:
        val=open_store(out,'budget_val_v02')
        extra=ridge(out,c,b,val)
        selection=choose(out,val,b,additional_store=extra)
        print('Selection frozen:',selection['methods'],flush=True)
    if args.command in ['all','main']:
        selection=json.loads((out/'configs/selected_v02.json').read_text())
        store=main_store(out,c,selection);counts=store.run()
        print('Test invocation:',counts,'(cache is not independent reproduction)',flush=True)
    if args.command in ['all','verify']:
        from verify_release import verify_all
        verify_all(out,ROOT)
    if args.command in ['all','summarize']:
        from release_analysis import summarize
        summarize(out,ROOT)

if __name__=='__main__':main()
