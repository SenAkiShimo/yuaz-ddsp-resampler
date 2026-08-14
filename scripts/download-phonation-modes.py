#!/usr/bin/env python3
import argparse,json,os,re,time,urllib.request,zipfile
from pathlib import Path
from resumable_http import download_any,remote_size,human_bytes

FILE_GUID='cwquj'
PROJECT='pa3ha'
NAME='ND357A_24bit_cut_ALL.zip'
API=os.environ.get('YUAZ_OSF_FILE_API',f'https://api.osf.io/v2/files/{FILE_GUID}/')
FALLBACKS=tuple(x for x in [os.environ.get('YUAZ_OSF_DOWNLOAD_URL'),f'https://osf.io/download/{FILE_GUID}/',f'https://osf.io/{FILE_GUID}/download'] if x)
UA='Yuaz-DDSP-Resampler/0.2.8ai.13-phonation-training'
AUDIO_EXT={'.wav','.flac','.aif','.aiff'}


def get_json(url):
    req=urllib.request.Request(url,headers={'User-Agent':UA,'Accept':'application/json'})
    with urllib.request.urlopen(req,timeout=45) as r:return json.loads(r.read().decode('utf-8'))


def metadata():
    urls=[];size=0;name=NAME
    try:
        data=get_json(API).get('data',{});attrs=data.get('attributes',{}) or {};links=data.get('links',{}) or {}
        name=str(attrs.get('name') or NAME);size=int(attrs.get('size') or 0)
        if links.get('download'):urls.append(str(links['download']))
    except Exception as exc:
        print(f'OSF metadata API unavailable ({exc}); using stable file-GUID download URL.',flush=True)
    urls.extend(FALLBACKS)
    # de-duplicate while preserving priority
    return name,size,list(dict.fromkeys(urls))


def quality_from_path(path):
    t=str(path).lower()
    checks=(('breathy',r'(^|[^a-z])breathy([^a-z]|$)'),('modal',r'(^|[^a-z])(neutral|modal)([^a-z]|$)'),('flow',r'(^|[^a-z])flow([^a-z]|$)'),('pressed',r'(^|[^a-z])pressed(?!ta)([^a-z]|$)'),('pressedta',r'(^|[^a-z])pressedta([^a-z]|$)'))
    for q,pat in checks:
        if re.search(pat,t):return q
    return None


def pair_key(path):
    t=Path(path).as_posix().lower()
    t=re.sub(r'(^|[/_. -])(breathy|neutral|modal|flow|pressedta|pressed)(?=($|[/_. -]))',' ',t)
    t=re.sub(r'\.(wav|flac|aiff?|wave)$','',t)
    return re.sub(r'[^a-z0-9]+',' ',t).strip()


def validate_extract(zip_path,root):
    with zipfile.ZipFile(zip_path,'r') as z:
        bad=z.testzip()
        if bad:raise RuntimeError(f'OSF phonation ZIP CRC failure: {bad}')
        infos=[i for i in z.infolist() if not i.is_dir()]
        audio=[i for i in infos if Path(i.filename).suffix.lower() in AUDIO_EXT]
        if len(audio)<100:raise RuntimeError(f'OSF phonation archive looks incomplete: only {len(audio)} audio files')
        counts={q:0 for q in ('breathy','modal','flow','pressed','pressedta')}
        for i in audio:
            q=quality_from_path(i.filename)
            if q:counts[q]+=1
        if min(counts['breathy'],counts['modal'],counts['pressed'])<10:
            raise RuntimeError(f'OSF phonation labels not recovered from archive paths: {counts}')
        extracted=root/'extracted';marker=root/'.extracted-ok.json'
        current={'archive_bytes':zip_path.stat().st_size,'archive_mtime_ns':zip_path.stat().st_mtime_ns}
        need=True
        if marker.is_file():
            try:need=json.loads(marker.read_text())!=current
            except Exception:need=True
        if need:
            import shutil
            if extracted.exists():shutil.rmtree(extracted)
            extracted.mkdir(parents=True,exist_ok=True)
            z.extractall(extracted)
            marker.write_text(json.dumps(current,indent=2)+'\n')
    files=[]
    for p in sorted((root/'extracted').rglob('*')):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXT:continue
        rel=p.relative_to(root).as_posix();q=quality_from_path(rel)
        files.append({'path':rel,'quality':q,'pair_key':pair_key(rel),'size':p.stat().st_size})
    counts={q:sum(1 for x in files if x['quality']==q) for q in ('breathy','modal','flow','pressed','pressedta')}
    groups={}
    for x in files:
        if x['quality'] in ('breathy','modal','pressed') and x['pair_key']:
            groups.setdefault(x['pair_key'],set()).add(x['quality'])
    complete=sum(1 for v in groups.values() if {'breathy','modal','pressed'}<=v)
    if complete<4:
        raise RuntimeError(f'Archive is valid but only {complete} direct breathy/modal/pressed triplets were paired. Counts={counts}')
    return files,counts,complete


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--local-dir',required=True);ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
    root=Path(a.local_dir).expanduser().resolve();raw=root/'raw';raw.mkdir(parents=True,exist_ok=True);dest=raw/NAME
    name,size,urls=metadata()
    if not size:
        for u in urls:
            try:
                size=remote_size(u,UA,timeout=30)
                if size:break
            except Exception:pass
    part=dest.with_name(dest.name+'.part');have=0
    if dest.is_file() and (not size or dest.stat().st_size==size):have=dest.stat().st_size
    elif part.is_file():have=part.stat().st_size
    print('Phonation Modes source: OSF project',PROJECT,'file',FILE_GUID)
    print('Archive:',name)
    print('Destination:',root)
    print('Known total:',human_bytes(size) if size else 'unknown until server responds')
    print('Reusable bytes:',human_bytes(have))
    if size:print('Known remaining:',human_bytes(max(0,size-have)))
    if a.dry_run:return
    if not download_any(urls,dest,size,UA,retries=120,timeout=60,label=NAME):raise SystemExit(130)
    try:
        files,counts,complete=validate_extract(dest,root)
    except Exception:
        # Do not ever leave a false "ready" dataset. Preserve bytes as a resumable
        # partial so another run can repair/replace it.
        bad=dest.with_name(dest.name+'.part')
        if dest.exists() and not bad.exists():dest.replace(bad)
        raise
    manifest={'format':1,'source':'OSF Phonation Modes Dataset','project':PROJECT,'file_guid':FILE_GUID,'archive':NAME,'files':files,'quality_counts':counts,'direct_triplets':complete,'created_at':time.time()}
    (root/'.yuaz-phonation-modes-osf-manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print('Validated audio files:',len(files),'quality counts:',counts)
    print('Direct breathy/modal/pressed triplets:',complete)
    print('Phonation Modes ready:',root)

if __name__=='__main__':main()
