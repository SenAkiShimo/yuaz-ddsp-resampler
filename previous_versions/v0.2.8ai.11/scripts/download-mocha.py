#!/usr/bin/env python3
import argparse,json,os,tarfile,time
from pathlib import Path
from resumable_http import download_any,remote_size,human_bytes

HTTPS=os.environ.get('YUAZ_MOCHA_BASE_HTTPS','https://data.cstr.ed.ac.uk/mocha').rstrip('/')
HTTP=os.environ.get('YUAZ_MOCHA_BASE_HTTP','http://data.cstr.ed.ac.uk/mocha').rstrip('/')
FILES=['fsew0_v1.1.tar.gz','msak0_v1.1.tar.gz','README_v1.2.txt','LICENCE.txt']
UA='Yuaz-DDSP-Resampler/0.2.8ai.11-mocha-training'


def tar_valid(path):
    try:
        with tarfile.open(path,'r:gz') as tf:
            members=tf.getmembers()
            return len(members)>100 and any(m.name.endswith('.wav') for m in members) and any(m.name.endswith('.lar') for m in members) and any(m.name.endswith('.ema') for m in members)
    except Exception:return False


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--local-dir',required=True);ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
    root=Path(a.local_dir).expanduser().resolve();raw=root/'raw';raw.mkdir(parents=True,exist_ok=True)
    manifest=root/'.yuaz-mocha-core-manifest.json';cached={}
    if manifest.is_file():
        try:cached={x['name']:x for x in json.loads(manifest.read_text()).get('files',[])}
        except Exception:cached={}
    items=[]
    for name in FILES:
        urls=[f'{HTTPS}/{name}',f'{HTTP}/{name}'];size=int(cached.get(name,{}).get('size') or 0)
        if not size:
            for url in urls:
                try:
                    size=remote_size(url,UA,timeout=20)
                    if size:break
                except Exception:pass
        items.append({'name':name,'urls':urls,'url':urls[0],'size':size})
    manifest.write_text(json.dumps({'format':2,'source':'MOCHA-TIMIT CSTR official','files':items,'created_at':time.time()},indent=2)+'\n')
    known=sum(x['size'] for x in items);have=0;unknown=0
    for x in items:
        d=raw/x['name'];p=d.with_name(d.name+'.part');sz=x['size']
        if not sz:unknown+=1
        if d.exists() and (not sz or d.stat().st_size==sz):have+=d.stat().st_size
        elif p.exists():have+=min(p.stat().st_size,sz or p.stat().st_size)
    print(f'MOCHA core destination: {root}')
    print(f'Known total: {human_bytes(known)}'+(f' + {unknown} unknown-size file(s)' if unknown else ''))
    print(f'Reusable bytes: {human_bytes(have)}')
    print(f'Known remaining: {human_bytes(max(0,known-have))}')
    if a.dry_run:return
    for x in items:
        dest=raw/x['name']
        # If a previous run moved a truncated archive into the final name before
        # validation existed, put it back into the resumable .part path.
        if x['name'].endswith('.tar.gz') and dest.exists() and not tar_valid(dest):
            part=dest.with_name(dest.name+'.part')
            if not part.exists():os.replace(dest,part)
            else:dest.unlink()
        if not download_any(x['urls'],dest,x['size'],UA,retries=120,timeout=60,label=x['name']):raise SystemExit(130)
        if x['name'].endswith('.tar.gz') and not tar_valid(dest):
            part=dest.with_name(dest.name+'.part')
            if part.exists():part.unlink()
            os.replace(dest,part)
            raise RuntimeError(f'{x["name"]} download ended but archive validation failed; bytes preserved as .part for resume.')
    extracted=root/'extracted';extracted.mkdir(parents=True,exist_ok=True)
    for name in ('fsew0_v1.1.tar.gz','msak0_v1.1.tar.gz'):
        marker=root/(name+'.extracted.json');src=raw/name;sig={'bytes':src.stat().st_size,'mtime_ns':src.stat().st_mtime_ns}
        same=False
        if marker.is_file():
            try:same=json.loads(marker.read_text())==sig
            except Exception:pass
        if same:continue
        print('Extracting',name)
        with tarfile.open(src,'r:gz') as tf:tf.extractall(extracted,filter='data')
        marker.write_text(json.dumps(sig,indent=2)+'\n')
    # Keep the upstream licence visibly with the prepared corpus as required by it.
    if (raw/'LICENCE.txt').is_file():(root/'LICENCE.txt').write_bytes((raw/'LICENCE.txt').read_bytes())
    wavs=list(extracted.rglob('*.wav'));lars=list(extracted.rglob('*.lar'));emas=list(extracted.rglob('*.ema'))
    if min(len(wavs),len(lars),len(emas))<100:raise RuntimeError(f'MOCHA extraction incomplete: wav={len(wavs)} lar={len(lars)} ema={len(emas)}')
    ready={'format':1,'wav':len(wavs),'lar':len(lars),'ema':len(emas),'created_at':time.time()}
    (root/'.yuaz-mocha-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
    print(f'MOCHA core ready: {root} (wav={len(wavs)}, lar={len(lars)}, ema={len(emas)})')

if __name__=='__main__':main()
