#!/bin/bash
set -euo pipefail
DST_ROOT="$HOME/Library/Application Support/YuazDDSP/0.2.8ai.16"
PY="$DST_ROOT/.venv/bin/python"
LOG="$DST_ROOT/logs/render_requests.jsonl"
[ -x "$PY" ] || { echo "Installed ai16 Python missing: $PY" >&2; exit 1; }
[ -f "$LOG" ] || { echo "Render log missing: $LOG" >&2; exit 1; }
"$PY" - "$LOG" <<'PY'
import hashlib,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
from importlib.util import spec_from_file_location,module_from_spec

log=Path(sys.argv[1]).resolve()
controls_path=Path.home()/"Library/Application Support/YuazDDSP/0.2.8ai.16/src/yuaz_ddsp_resampler/controls.py"
spec=spec_from_file_location("yuaz_controls",controls_path); mod=module_from_spec(spec); spec.loader.exec_module(mod)
rows=[]
with log.open('r',encoding='utf-8') as f:
    for line in f:
        try: rows.append(json.loads(line))
        except Exception: pass

def latest(target):
    for row in reversed(rows):
        req=row.get('request') or {}
        c=mod.parse_yuaz_controls(str(req.get('flags') or ''))
        if float(c.voicing)==float(target):
            p=Path(str(req.get('output') or '')).expanduser()
            if p.is_file(): return row,p
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
        print(json.dumps({'type':'missing','yv':v},separators=(',',':'))); continue
    y,sr=read(p); found[v]=(row,p,y,sr)
    print(json.dumps({
        'type':'actual_openutau_cache','yv':v,
        'flags':str((row.get('request') or {}).get('flags') or ''),
        'path':str(p),'bytes':p.stat().st_size,
        'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
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
    same=p0.read_bytes()==p1.read_bytes()
    d=rms(found[100][2][:min(len(found[0][2]),len(found[100][2]))]-found[0][2][:min(len(found[0][2]),len(found[100][2]))])
    if same or d < 1e-7:
        print('DIAGNOSIS: actual-resampler-cache-identical')
    else:
        print('DIAGNOSIS: resampler-cache-different; playback/phrase-cache-above-resampler')
        print('PLAY_YV0=' + str(p0))
        print('PLAY_YV100=' + str(p1))
PY
