#!/usr/bin/env python3
import os, time, urllib.request
from pathlib import Path


def human_bytes(n):
    n=float(n or 0)
    for u in ('B','KiB','MiB','GiB','TiB'):
        if n < 1024 or u == 'TiB':
            return f'{n:.1f} {u}'
        n /= 1024


def remote_size(url, user_agent='Yuaz-DDSP-Resampler/0.2.8ai.11', timeout=30):
    headers={'User-Agent':user_agent,'Range':'bytes=0-0','Accept':'*/*'}
    req=urllib.request.Request(url,headers=headers)
    with urllib.request.urlopen(req,timeout=timeout) as r:
        cr=r.headers.get('Content-Range','')
        if '/' in cr:
            try:return int(cr.rsplit('/',1)[1])
            except Exception:pass
        cl=r.headers.get('Content-Length')
        return int(cl) if cl else 0


def _response_total(response, existing, status):
    cr=response.headers.get('Content-Range','')
    if '/' in cr:
        try:
            total=int(cr.rsplit('/',1)[1])
            if total > 0:return total
        except Exception:pass
    cl=response.headers.get('Content-Length')
    if cl:
        try:
            n=int(cl)
            return (existing+n) if status==206 and existing else n
        except Exception:pass
    return 0


def download_any(urls, dest, expected=0, user_agent='Yuaz-DDSP-Resampler/0.2.8ai.11', retries=120, timeout=60, label=None):
    if isinstance(urls,str):urls=[urls]
    urls=[u for u in urls if u]
    if not urls:raise ValueError('No download URL supplied')
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);part=dest.with_name(dest.name+'.part')
    if dest.is_file() and (not expected or dest.stat().st_size==expected):return True
    if dest.exists() and expected and dest.stat().st_size!=expected:
        if not part.exists():os.replace(dest,part)
        else:dest.unlink()
    delay=1.0
    learned_total=int(expected or 0)
    for attempt in range(1,retries+1):
        url=urls[(attempt-1)%len(urls)]
        existing=part.stat().st_size if part.exists() else 0
        if learned_total and existing>learned_total:
            part.unlink();existing=0
        h={'User-Agent':user_agent,'Accept':'*/*'}
        if existing:h['Range']=f'bytes={existing}-'
        try:
            req=urllib.request.Request(url,headers=h)
            with urllib.request.urlopen(req,timeout=timeout) as r:
                status=getattr(r,'status',r.getcode())
                append=bool(existing and status==206)
                if existing and not append:
                    existing=0
                total=_response_total(r,existing if append else 0,status)
                if total:learned_total=total
                mode='ab' if append else 'wb';last=time.time();lastn=existing
                with part.open(mode) as f:
                    while True:
                        b=r.read(1024*1024)
                        if not b:break
                        f.write(b)
                        now=time.time()
                        if now-last>=0.5:
                            got=f.tell();speed=max(0,(got-lastn)/(now-last));pct=(100*got/learned_total) if learned_total else 0
                            suffix=f'/{human_bytes(learned_total)}' if learned_total else ''
                            print(f'\r[{pct:6.2f}%] {label or dest.name} | {human_bytes(got)}{suffix} | {human_bytes(speed)}/s   ',end='',flush=True)
                            last,lastn=now,got
                got=part.stat().st_size
                if learned_total and got!=learned_total:raise IOError(f'size mismatch {got}!={learned_total}')
                os.replace(part,dest)
                print(f'\r[100.00%] {label or dest.name} | {human_bytes(got)} complete                 ')
                return True
        except KeyboardInterrupt:
            print('\nInterrupted; .part preserved for resume.')
            return False
        except Exception as e:
            if attempt==retries:raise RuntimeError(f'Failed after {retries} resumable attempts; .part preserved. Last URL {url}: {e}')
            print(f'\nNetwork retry {attempt}/{retries}: {e}')
            time.sleep(delay);delay=min(delay*1.45,30)
    return False


def download(url,dest,expected=0,user_agent='Yuaz-DDSP-Resampler/0.2.8ai.11',retries=120,timeout=60,label=None):
    return download_any([url],dest,expected,user_agent,retries,timeout,label)
