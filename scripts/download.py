"""Download the official original archive, resume safely, CRC-check, record provenance."""
from pathlib import Path
import datetime, hashlib, json, time, zipfile
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
FILE_ID = '1_EHrS-Sp25nzmbJKbwnNVzbSR57jb6_8'
URL = 'https://drive.usercontent.google.com/download'

def main():
    (ROOT/'reports').mkdir(parents=True,exist_ok=True)
    dest = ROOT / 'data/raw/RadioMapSeer.zip'
    dest.parent.mkdir(parents=True, exist_ok=True)
    provenance = ROOT/'reports/download.json'
    if provenance.exists() and dest.exists():
        old = json.loads(provenance.read_text())
        if dest.stat().st_size == old['bytes']:
            h = hashlib.sha256()
            with dest.open('rb') as f:
                for block in iter(lambda: f.read(8*1024*1024), b''):
                    h.update(block)
            if h.hexdigest() == old['sha256']:
                print('Previously CRC-verified archive SHA-256 matches; no download needed', flush=True)
                return
        raise RuntimeError('Existing archive differs from recorded provenance; move it aside before retrying')
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    session = requests.Session()
    for attempt in range(8):
        try:
            params = dict(id=FILE_ID, export='download')
            r = session.get(URL, params=params, timeout=(30, 90), stream=True)
            r.raise_for_status()
            if 'text/html' in r.headers.get('Content-Type', ''):
                form = BeautifulSoup(r.text, 'html.parser').find('form')
                if form is None:
                    raise RuntimeError('Drive did not provide a download form')
                params = {i['name']: i.get('value', '') for i in form.find_all('input') if i.get('name')}
                r.close()
            else:
                r.close()
            offset = dest.stat().st_size if dest.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            r = session.get(URL, params=params, headers=headers, timeout=(30, 90), stream=True)
            r.raise_for_status()
            if 'text/html' in r.headers.get('Content-Type', ''):
                raise RuntimeError('Expected ZIP, received HTML: ' + r.text[:250])
            if offset and r.status_code == 206:
                if not r.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Incorrect resumed range')
                mode = 'ab'
            else:
                mode = 'wb'  # server ignored Range: restart, never append a full file
                offset = 0
            print('Downloading', r.status_code, r.headers.get('Content-Length'), 'offset', offset, flush=True)
            last = time.monotonic()
            with dest.open(mode) as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
                    if time.monotonic() - last > 15:
                        print(f'{f.tell()/1e6:.1f} MB downloaded', flush=True)
                        last = time.monotonic()
            with zipfile.ZipFile(dest) as z:
                bad = z.testzip()
                if bad:
                    raise RuntimeError('CRC failure: ' + bad)
                inventory = [dict(path=i.filename, size=i.file_size, crc=i.CRC) for i in z.infolist()]
            h = hashlib.sha256()
            with dest.open('rb') as f:
                for block in iter(lambda: f.read(8*1024*1024), b''):
                    h.update(block)
            meta = dict(official_page='https://radiomapseer.github.io/index.html',
                        source=f'https://drive.google.com/file/d/{FILE_ID}/view',
                        started_utc=started, completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        bytes=dest.stat().st_size, sha256=h.hexdigest(), crc_checked=True,
                        members=len(inventory))
            (ROOT/'reports/download.json').write_text(json.dumps(meta, indent=2))
            (ROOT/'reports/archive_inventory.json').write_text(json.dumps(inventory, indent=2))
            dest.chmod(0o444)
            print(meta, flush=True)
            return
        except Exception as e:
            print(f'Attempt {attempt+1}: {type(e).__name__}: {e}', flush=True)
            if attempt == 7:
                raise
            time.sleep(2)

if __name__ == '__main__':
    main()
