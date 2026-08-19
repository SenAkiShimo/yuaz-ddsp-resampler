#!/bin/bash
set -euo pipefail
DST_ROOT="$HOME/Library/Application Support/YuazDDSP/0.2.8ai.16"
PY="$DST_ROOT/.venv/bin/python"
LOG="$DST_ROOT/logs/render_requests.jsonl"
OU_CACHE="$HOME/Library/Caches/OpenUtau"
[ -x "$PY" ] || { echo "Installed ai16 Python missing: $PY" >&2; exit 1; }
[ -f "$LOG" ] || { echo "Render log missing: $LOG" >&2; exit 1; }
[ -d "$OU_CACHE" ] || { echo "OpenUtau cache missing: $OU_CACHE" >&2; exit 1; }
"$PY" - "$LOG" "$OU_CACHE" <<'PY'
import hashlib,json,re,sys
from pathlib import Path
import numpy as np
import soundfile as sf
from importlib.util import spec_from_file_location,module_from_spec

log=Path(sys.argv[1]).resolve()
ou_cache=Path(sys.argv[2]).resolve()
controls_path=Path.home()/"Library/Application Support/YuazDDSP/0.2.8ai.16/src/yuaz_ddsp_resampler/controls.py"
spec=spec_from_file_location("yuaz_controls",controls_path); mod=module_from_spec(spec); spec.loader.exec_module(mod)
rows=[]
with log.open('r',encoding='utf-8') as f:
    for line in f:
        try: rows.append(json.loads(line))
        except Exception: pass

def under_openutau_cache(path):
    try:
        path.resolve().relative_to(ou_cache)
        return True
    except Exception:
        return False

def normalized_flags(flags):
    return re.sub(r'YV[+-]?(?:\d+(?:\.\d*)?|\.\d+)','YV*',str(flags or ''),flags=re.I)

def candidate_rows():
    out=[]
    for idx,row in enumerate(rows):
        req=row.get('request') or {}
        p=Path(str(req.get('output') or '')).expanduser()
        if not p.is_file() or not under_openutau_cache(p):
            continue
        flags=str(req.get('flags') or '')
        c=mod.parse_yuaz_controls(flags)
        out.append((idx,row,p,float(c.voicing),str(req.get('input') or ''),normalized_flags(flags)))
    return out

cands=candidate_rows()
if not cands:
    raise RuntimeError('No Yuaz render outputs were found inside the OpenUtau cache directory')

# Pick the newest real OpenUtau render as the anchor, then compare only requests
# for the same source input and same non-YV flag pattern. This excludes our /tmp
# direct-engine diagnostics and avoids mixing unrelated phrases.
anchor=cands[-1]
anchor_input=anchor[4]
anchor_flags=anchor[5]
group=[x for x in cands if x[4]==anchor_input and x[5]==anchor_flags]

print(json.dumps({
    'type':'anchor','openutau_cache':str(ou_cache),'group_size':len(group),
    'input':anchor_input,'normalized_flags':anchor_flags,
},ensure_ascii=False,separators=(',',':')))

def latest(target):
    for item in reversed(group):
        if item[3]==float(target):
            return item[1],item[2]
    return None,None

def read(path):
    y,sr=sf.read(path,always_2d=False)
    if getattr(y,'ndim',1)>1: y=np.mean(y,axis=1)
    return np.asarray(y,dtype=np.float64),int(sr)

def rms(x): return float(np.sqrt(np.mean(x*x)+1e-15))

found={}
for v in (0,100,-100):
    row,p=latest(v)
    if p is None:
        print(json.dumps({'type':'missing','yv':v,'reason':'no matching real OpenUtau cache render in anchor group'},separators=(',',':')))
        continue
    y,sr=read(p); found[v]=(row,p,y,sr)
    print(json.dumps({
        'type':'actual_openutau_cache','yv':v,
        'flags':str((row.get('request') or {}).get('flags') or ''),
        'path':str(p),'inside_openutau_cache':under_openutau_cache(p),
        'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'samples':len(y),'sample_rate':sr,'rms':rms(y),
        'render_total_ms':(row.get('result') or {}).get('yuaz_render_total_ms'),
    },ensure_ascii=False,separators=(',',':')))

for a,b in ((0,100),(0,-100),(-100,100)):
    if a not in found or b not in found: continue
    ya=found[a][2]; yb=found[b][2]; n=min(len(ya),len(yb)); ya=ya[:n]; yb=yb[:n]
    d=rms(yb-ya); base=max(1e-12,.5*(rms(ya)+rms(yb)))
    print(json.dumps({
        'type':'actual_cache_compare','a':a,'b':b,
        'byte_identical':found[a][1].read_bytes()==found[b][1].read_bytes(),
        'difference_rms':d,'difference_ratio':d/base,
    },separators=(',',':')))

if 0 in found and 100 in found:
    p0=found[0][1]; p1=found[100][1]
    n=min(len(found[0][2]),len(found[100][2]))
    same=p0.read_bytes()==p1.read_bytes()
    d=rms(found[100][2][:n]-found[0][2][:n])
    base=max(1e-12,.5*(rms(found[0][2][:n])+rms(found[100][2][:n])))
    if same or d/base < 1e-4:
        print('DIAGNOSIS: real-openutau-resampler-cache-identical')
    else:
        print('DIAGNOSIS: real-openutau-resampler-cache-different')
        print('PLAY_YV0=' + str(p0))
        print('PLAY_YV100=' + str(p1))
else:
    print('DIAGNOSIS: incomplete-real-openutau-cache-group')
PY
