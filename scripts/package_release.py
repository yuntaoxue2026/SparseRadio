"""Curated local/public artifacts; inspect links, secrets and package categories."""
from pathlib import Path
import hashlib,json,re,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/v02'
RELEASE=ROOT/'releases'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def addtree(z,folder,prefix):
    for p in folder.rglob('*'):
        if p.is_file() and not set(p.parts)&{'__pycache__','.pytest_cache','mpl-cache'} and not p.name.startswith('._'):
            z.write(p,prefix+'/'+str(p.relative_to(folder)))

def main():
    RELEASE.mkdir(exist_ok=True)
    public=RELEASE/'public_candidate'
    if public.exists():shutil.rmtree(public)
    public.mkdir()
    def copy(src,dest=None):
        src=ROOT/src;dest=public/(dest if dest else str(src.relative_to(ROOT)))
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    for n in ['README.md','LICENSE','CITATION.cff','pyproject.toml','requirements-lock.txt','CHANGELOG.md','DATA_LICENSE.md','FIGURE_ATTRIBUTION.md','.gitignore']:
        copy(n)
    for folder in ['src/sparseradiomap','tests']:
        for p in (ROOT/folder).glob('*.py'):
            if folder=='src/sparseradiomap' and p.name in ['analysis.py','experiment.py']:continue
            copy(p.relative_to(ROOT))
    for n in ['pipeline.py','download.py','verify_release.py','release_analysis.py','package_release.py']:
        copy('scripts/'+n)
    for p in (ROOT/'configs').glob('*.json'):copy(p.relative_to(ROOT))
    for n in ['manifest.csv']:copy('data/'+n)
    for n in ['main.csv','smoke.csv','summary.csv','case_selection.json']:
        copy('results/'+n)
    copy('reports/download.json')
    for p in (OUT/'configs').glob('*.json'):copy(p.relative_to(ROOT))
    for p in (OUT/'results').glob('*.csv'):copy(p.relative_to(ROOT))
    for version in ['budget_dev_v02','budget_val_v02','test_v02']:
        for n in ['results.csv','run_manifest.json','training_curves.csv']:
            p=OUT/'results'/version/n
            if p.exists():copy(p.relative_to(ROOT))
    for p in (OUT/'figures').glob('*'):
        if p.suffix in ['.png','.svg']:copy(p.relative_to(ROOT))
    safe=['report_zh.md','budget_candidate_scores.csv','diagnostic_budget.json','further_tuning_gate.json',
          'score_verification.json','baseline_v01_recomputed.csv','test_v02_recomputed.csv','clean_environment.json',
          'requirements-resolved-clean.txt','tests.log','clean_tests.log','trajectory_200_equivalence.json',
          'processing.json','presentation_manifest.json','DATA_LICENSE.md','FIGURE_ATTRIBUTION.md','experiment_ledger.json','baseline_integrity.json']
    safe += ['public_tests.log','public_candidate_fresh_verification.json']
    safe += ['release_source_manifest.json']
    for n in safe:
        p=OUT/'reports'/n
        if p.exists():copy(p.relative_to(ROOT))
    for split in ['dev','test']:
        path=ROOT/f'experiments/fresh_{split}_v02/reports/fresh_verification.json'
        if path.exists():copy(path.relative_to(ROOT))
    path=ROOT/'experiments/fresh_explicit_cpu_v02/reports/fresh_verification.json'
    if path.exists():copy(path.relative_to(ROOT))
    copy('experiments/v02/reports/license_evidence/sources.json')
    (public/'experiments/v02/reports/license_evidence/EXCERPT.md').write_text(
        'Official source: https://raw.githubusercontent.com/radiomapseer/radiomapseer.github.io/master/index.html\n\n'
        'The original dataset download is immediately followed by `rel="license" href="http://creativecommons.org/licenses/by/4.0/"`.\n'
        'See source URLs and SHA-256 in sources.json. Full HTML evidence is retained locally.\n')
    # All displayed numerical/source files must be portable and credential-free.
    forbidden=[r'(?:AKIA|ASIA)[A-Z0-9]{16}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'gh[pousr]_[A-Za-z0-9]{30,}',r'/Users/',r'/Volumes/',r'/opt/anaconda']
    scan_errors=[];broken=[]
    text_suffixes={'.md','.json','.csv','.txt','.toml','.py','.cff','.log'}
    for p in public.rglob('*'):
        if not p.is_file() or p.suffix not in text_suffixes:continue
        text=p.read_text(errors='replace')
        # Test fixtures/regex code may contain the pattern itself, but no matched secrets.
        if p.name!='package_release.py':
            for pattern in forbidden:
                if re.search(pattern,text):scan_errors.append(str(p.relative_to(public))+': '+pattern)
        if p.suffix=='.md':
            for target in re.findall(r'\]\(([^)]+)\)',text):
                target=target.strip('<>').split('#')[0]
                if target and not re.match(r'\w+://',target) and not (p.parent/target).exists():broken.append(str(p.relative_to(public))+': '+target)
    if scan_errors or broken:raise ValueError(json.dumps(dict(portability_or_secret_scan=scan_errors,broken_links=broken),indent=2))
    public_zip=RELEASE/'SparseRadioMap_v01_public_candidate.zip'
    with zipfile.ZipFile(public_zip,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:addtree(z,public,'SparseRadioMap')
    local_zip=RELEASE/'SparseRadioMap_v02_complete_local.zip'
    with zipfile.ZipFile(local_zip,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        addtree(z,public,'SparseRadioMap')
        # Replace curated numerical duplicates by identical complete originals where needed.
        existing=set(z.namelist())
        for folder in ['experiments','history','results','reports','figures','data/processed']:
            for p in (ROOT/folder).rglob('*'):
                if not p.is_file() or set(p.parts)&{'__pycache__','.pytest_cache','mpl-cache'} or p.name.startswith('._') or p.suffix=='.html':continue
                name='SparseRadioMap/'+str(p.relative_to(ROOT))
                if name not in existing:z.write(p,name);existing.add(name)
        z.writestr('SparseRadioMap/LOCAL_PACKAGE.md',
          'Complete derived arrays, version records, trajectories and baseline snapshot are included.\n'
          'Virtual environments/caches are excluded. Original RadioMapSeer.zip remains an external read-only input,\n'
          'identified by SHA-256 in reports/download.json; reuse the existing ZIP or run scripts/download.py.\n')
    artifacts=[]
    for p in [public_zip,local_zip]:
        with zipfile.ZipFile(p) as z:
            if z.testzip() is not None:raise ValueError('Package CRC failure')
            count=len(z.namelist())
        artifacts.append(dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p),files=count))
    (RELEASE/'package_verification.json').write_text(json.dumps(dict(artifacts=artifacts,credential_scan='passed',unrelated_path_scan='passed',markdown_link_check='passed',published=False),indent=2))
    print(json.dumps(artifacts,indent=2))

if __name__=='__main__':main()
