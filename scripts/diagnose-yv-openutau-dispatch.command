#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST_ROOT="$HOME/Library/Application Support/YuazDDSP/0.2.8ai.16"
PY="$DST_ROOT/.venv/bin/python"
LOG="$DST_ROOT/logs/render_requests.jsonl"
[ -x "$PY" ] || { echo "Installed ai16 Python missing: $PY" >&2; exit 1; }
[ -f "$LOG" ] || { echo "No render request log yet: $LOG" >&2; exit 1; }
export PYTHONPATH="$DST_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$DST_ROOT" "$LOG" <<'PY'
import hashlib,json,re,sys,time
from pathlib import Path
import numpy as np
import soundfile as sf

root=Path(sys.argv[1]).resolve(); log=Path(sys.argv[2]).resolve()
from yuaz_ddsp_resampler.client import load_config,ping,_status_matches,start_server,send
from yuaz_ddsp_resampler.controls import parse_yuaz_controls

lines=[]
with log.open('r',encoding='utf-8') as f:
    for line in f:
        try: lines.append(json.loads(line))
        except Exception: pass
if not lines: raise RuntimeError('render_requests.jsonl contains no valid requests')
recent=lines[-40:]
print(json.dumps({'type':'recent_log','count':len(recent),'log':str(log)},ensure_ascii=False,separators=(',',':')))
for i,row in enumerate(recent[-12:],1):
    req=row.get('request') or {}; res=row.get('result') or {}
    flags=str(req.get('flags') or '')
    c=parse_yuaz_controls(flags)
    print(json.dumps({
        'type':'recent_request','index':i,'flags':flags,'parsed_yv':float(c.voicing),
        'input':str(req.get('input') or ''),'output':str(req.get('output') or ''),
        'render_total_ms':res.get('yuaz_render_total_ms'),'ok':res.get('ok'),
        'runtime_route':((res.get('yuaz_ai_effects') or {}).get('phonation') or {}).get('runtime_route') if isinstance(res.get('yuaz_ai_effects'),dict) else None,
    },ensure_ascii=False,separators=(',',':')))

base=None
for row in reversed(lines):
    req=row.get('request') or {}
    p=Path(str(req.get('input') or '')).expanduser()
    if p.is_file():
        base=dict(req); break
if base is None: raise RuntimeError('No recent render request points to an existing input WAV')

config,config_path=load_config(root)
host=config.get('host','127.0.0.1'); port=int(config.get('port',47888)); runtime_id=str(config.get('runtime_id') or '0.2.8ai.16')
status=ping(host,port)
if not _status_matches(status,runtime_id,root):
    start_server(root,config_path,host,port,runtime_id)

flags0=re.sub(r'YV[+-]?(?:\d+(?:\.\d*)?|\.\d+)','',str(base.get('flags') or ''),flags=re.I)+'YV0'
flags1=re.sub(r'YV[+-]?(?:\d+(?:\.\d*)?|\.\d+)','',str(base.get('flags') or ''),flags=re.I)+'YV100'
stamp=str(int(time.time()))
out0=Path('/tmp')/f'yuaz-yv0-{stamp}.wav'; out1=Path('/tmp')/f'yuaz-yv100-{stamp}.wav'

def render(flags,out):
    req=dict(base); req['flags']=flags; req['output']=str(out)
    response=send(host,port,{'action':'render','request':req,'runtime_id':runtime_id},timeout=600)
    if not response.get('ok'): raise RuntimeError(response.get('error','Render failed'))
    return response

r0=render(flags0,out0); r1=render(flags1,out1)

def audio(path):
    y,sr=sf.read(path,always_2d=False)
    if getattr(y,'ndim',1)>1: y=np.mean(y,axis=1)
    return np.asarray(y,dtype=np.float64),int(sr)
y0,sr0=audio(out0); y1,sr1=audio(out1); n=min(len(y0),len(y1)); y0=y0[:n]; y1=y1[:n]
rms=lambda x: float(np.sqrt(np.mean(x*x)+1e-15))
diff=rms(y1-y0); denom=max(1e-12,0.5*(rms(y0)+rms(y1)))
sha0=hashlib.sha256(out0.read_bytes()).hexdigest(); sha1=hashlib.sha256(out1.read_bytes()).hexdigest()
print(json.dumps({
    'type':'direct_engine_test','base_input':str(base.get('input')),
    'flags_yv0':flags0,'flags_yv100':flags1,
    'parsed_yv0':float(parse_yuaz_controls(flags0).voicing),'parsed_yv100':float(parse_yuaz_controls(flags1).voicing),
    'output_yv0':str(out0),'output_yv100':str(out1),
    'sha256_yv0':sha0,'sha256_yv100':sha1,'byte_identical':sha0==sha1,
    'audio_rms_yv0':rms(y0),'audio_rms_yv100':rms(y1),'difference_rms':diff,'difference_ratio':diff/denom,
    'response_yv0_control':r0.get('yuaz_voicing'),'response_yv100_control':r1.get('yuaz_voicing'),
    'response_yv0_flags':r0.get('yuaz_flags_raw'),'response_yv100_flags':r1.get('yuaz_flags_raw'),
},ensure_ascii=False,separators=(',',':')))

if sha0==sha1 or diff/denom < 1e-4:
    print('DIAGNOSIS: engine-path-suspect')
else:
    print('DIAGNOSIS: engine-renders-YV; inspect OpenUtau dispatch/cache')
PY
