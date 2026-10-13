"""ZIP-backed DPM audit and deterministic region-level data selection."""
from pathlib import Path
from collections import Counter, defaultdict
from io import BytesIO
import csv, hashlib, json, re, zipfile
import numpy as np
from PIL import Image
from .core import decode_gray

def png(data):
    with Image.open(BytesIO(data)) as im:
        im.load()
        return np.asarray(im), im.mode

def audit(root, config):
    root = Path(root)
    # A real first run must not depend on scaffolding left by earlier tools.
    for directory in ['configs', 'reports', 'data/processed']:
        (root/directory).mkdir(parents=True, exist_ok=True)
    (root/'reports/audit_completed.json').unlink(missing_ok=True)
    archive = root/'data/raw/RadioMapSeer.zip'
    if not (root/'reports/download.json').exists():
        raise RuntimeError('Run downloader CRC/provenance verification first')
    metadata = json.loads((root/'reports/download.json').read_text())
    if archive.stat().st_size != metadata['bytes']:
        raise RuntimeError('Archive size changed')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        matches = [n for n in names if re.search(r'(?:^|/)gain/DPM/\d+_\d+\.png$', n)]
        if not matches:
            raise RuntimeError('No original DPM found')
        prefix = matches[0].split('gain/DPM/')[0]
        groups = defaultdict(list)
        for n in matches:
            m, t = map(int, Path(n).stem.split('_'))
            groups[m].append(t)
        # Select before inspecting signal values: fixed seed, no quality-based selection.
        rng = np.random.default_rng(config['selection_seed'])
        n_maps = sum(config['map_counts'].values())
        chosen = rng.choice(sorted(groups), n_maps, replace=False).tolist()
        rows = []
        offset = 0
        for split, count in config['map_counts'].items():
            for m in chosen[offset:offset+count]:
                for t in rng.choice(sorted(groups[m]), config['tx_per_map'], replace=False):
                    rows.append(dict(map_id=m, tx_id=int(t), split=split,
                                     gain_path=f'{prefix}gain/DPM/{m}_{t}.png',
                                     building_path=f'{prefix}png/buildings_complete/{m}.png',
                                     antenna_path=f'{prefix}png/antennas/{m}_{t}.png'))
            offset += count
        write_csv(root/'data/manifest.csv', rows)
        (root/'configs/split.json').write_text(json.dumps({s:chosen[sum(list(config['map_counts'].values())[:i]):sum(list(config['map_counts'].values())[:i+1])] for i,s in enumerate(config['map_counts'])},indent=2))
        # Decode/hash ALL DPM; audit complete original buildings and Tx PNGs too.
        audit_names = matches + [n for n in names if re.search(r'(?:^|/)png/(buildings_complete/\d+|antennas/\d+_\d+)\.png$', n)]
        counts = Counter()
        dpm_names = set(matches)
        modes, shapes, dtypes, values = Counter(), Counter(), Counter(), Counter()
        hashes = defaultdict(list)
        damaged = []
        stats = []
        selected = {r['gain_path']: r for r in rows}
        selected_paths = set()
        for r in rows:
            selected_paths.update(r[k] for k in ['gain_path','building_path','antenna_path'])
        zero_outdoor = zero_building = outdoor_total = building_total = 0
        bcache = {}
        tx_mismatch = 0
        coordinate_mismatch = 0
        tx_json_cache = {}
        for i,n in enumerate(audit_names):
            category = 'DPM' if n in dpm_names else ('buildings' if 'buildings_complete' in n else 'antennas')
            try:
                raw = z.read(n)
                a, mode = png(raw)
                sha = hashlib.sha256(raw).hexdigest()
                hashes[(category,sha)].append(n)
                counts[category] += 1
                modes[(category,mode)] += 1
                shapes[(category,str(a.shape))] += 1
                dtypes[(category,str(a.dtype))] += 1
                if a.shape != (256,256) or a.dtype != np.uint8 or mode != 'L':
                    raise ValueError(f'Unexpected image {mode} {a.shape} {a.dtype}')
                stat = dict(path=n, category=category, sha256=sha, bytes=len(raw),
                            dtype=str(a.dtype), mode=mode, height=a.shape[0], width=a.shape[1],
                            minimum=int(a.min()), maximum=int(a.max()), zeros=int((a==0).sum()))
                stats.append(stat)
                if category == 'buildings':
                    values.update({int(v):int(c) for v,c in zip(*np.unique(a,return_counts=True))})
                if category == 'DPM':
                    m = int(Path(n).stem.split('_')[0])
                    if m not in bcache:
                        bcache[m] = png(z.read(f'{prefix}png/buildings_complete/{m}.png'))[0]
                    b = bcache[m] > 0
                    zero_outdoor += int(((a==0)&~b).sum())
                    zero_building += int(((a==0)&b).sum())
                    outdoor_total += int((~b).sum())
                    building_total += int(b.sum())
                if category == 'antennas':
                    m, t = map(int, Path(n).stem.split('_'))
                    if m not in bcache:
                        bcache[m] = png(z.read(f'{prefix}png/buildings_complete/{m}.png'))[0]
                    if (a>0).sum() != 1 or np.any((a>0)&(bcache[m]>0)):
                        tx_mismatch += 1
                    if m not in tx_json_cache:
                        tx_json_cache[m] = json.loads(z.read(f'{prefix}antenna/{m}.json'))
                    x, y = tx_json_cache[m][t]
                    if not np.array_equal(np.argwhere(a>0), [[255-y,x]]):
                        coordinate_mismatch += 1
                if n in selected_paths:
                    p = root/'data/raw/selected'/n
                    p.parent.mkdir(parents=True,exist_ok=True)
                    p.write_bytes(raw)
                    p.chmod(0o444)
                if n in selected:
                    selected[n].update(sha256=sha, height=256,width=256,
                                       minimum=int(a.min()),maximum=int(a.max()))
            except Exception as e:
                damaged.append(dict(path=n,error=str(e)))
            if (i+1)%10000 == 0:
                print(f'Audited {i+1}/{len(audit_names)} PNGs',flush=True)
        write_csv(root/'reports/file_audit.csv', stats)
        write_csv(root/'data/manifest.csv', rows)
        duplicate_groups = [dict(category=k[0],sha256=k[1],paths=v) for k,v in hashes.items() if len(v)>1]
        (root/'reports/duplicates.json').write_text(json.dumps(duplicate_groups,indent=2))
        result = dict(prefix=prefix, archive_members=len(names), image_counts=dict(counts),
                      map_count=len(groups), tx_count_distribution=dict(Counter(map(len,groups.values()))),
                      missing_pairs=[r for r in rows if not all(r[k] in names for k in ['gain_path','building_path','antenna_path'])],
                      modes={str(k):v for k,v in modes.items()},shapes={str(k):v for k,v in shapes.items()},
                      dtypes={str(k):v for k,v in dtypes.items()},building_gray_counts=dict(values),
                      damaged=damaged,duplicate_groups=len(duplicate_groups),
                      zero_outdoor=zero_outdoor,outdoor_total=outdoor_total,
                      zero_building=zero_building,building_total=building_total,
                      antenna_mismatch=tx_mismatch,
                      coordinate_mismatch=coordinate_mismatch,
                      json_paths=[n for n in names if n.endswith('.json')],
                      license_paths=[n for n in names if re.search(r'license|readme|copying',n,re.I)])
        # Preserve author JSON metadata for coordinates verification without guessing.
        chosen_metadata = [f'{prefix}antenna/{m}.json' for m in chosen]
        chosen_metadata += [f'{prefix}polygon/buildings_complete/{m}.json' for m in chosen]
        chosen_metadata += [f'{prefix}dataset.csv']
        for n in chosen_metadata + result['license_paths']:
            if n not in names:
                continue
            if z.getinfo(n).file_size < 20_000_000:
                p=root/'reports/sources/archive_metadata'/n
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(z.read(n))
        (root/'reports/audit.json').write_text(json.dumps(result,indent=2))
        if damaged or result['missing_pairs']:
            raise RuntimeError('Audit failed; see audit.json')
        for r in rows:
            gain=png(z.read(r['gain_path']))[0]
            b=png(z.read(r['building_path']))[0]
            tx=png(z.read(r['antenna_path']))[0]
            p=root/'data/processed'/f"{r['map_id']}_{r['tx_id']}.npz"
            np.savez_compressed(p,y=decode_gray(gain).astype(np.float32),valid=b==0,buildings=b,antenna=tx)
        (root/'reports/audit_completed.json').write_text(json.dumps(dict(images=len(rows),archive_sha256=metadata['sha256'],selection_seed=config['selection_seed']),indent=2))
        print(json.dumps({k:v for k,v in result.items() if k not in ['json_paths','license_paths','missing_pairs']},indent=2),flush=True)

def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def read_manifest(root):
    with (Path(root)/'data/manifest.csv').open() as f:
        return [dict(r,map_id=int(r['map_id']),tx_id=int(r['tx_id'])) for r in csv.DictReader(f)]

def load_image(root,row):
    with np.load(Path(root)/'data/processed'/f"{row['map_id']}_{row['tx_id']}.npz") as f:
        return {k:f[k] for k in f.files}
