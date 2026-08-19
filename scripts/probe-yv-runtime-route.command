#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "Python not found: $PY" >&2; exit 1; }
INPUT="${1:-}"
if [ -z "$INPUT" ]; then read -r INPUT; fi
INPUT="${INPUT%/}"
WAV="$INPUT"
if [ -d "$INPUT" ]; then
  WAV="$(find "$INPUT" -type f -name 'additional.wav' -print | LC_ALL=C sort | head -n 1)"
  if [ -z "$WAV" ]; then WAV="$(find "$INPUT" -type f -iname '*.wav' -print | LC_ALL=C sort | head -n 1)"; fi
  [ -n "$WAV" ] || { echo "No WAV found under voicebank: $INPUT" >&2; exit 1; }
  echo "Selected WAV: $WAV"
elif [ ! -f "$WAV" ]; then
  echo "WAV or voicebank folder not found: $INPUT" >&2; exit 1
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$ROOT" "$WAV" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
import torch
from yuaz_ddsp_resampler.ai_control_training import NativeYuazDDSPExtractor
from yuaz_ddsp_resampler.ai_vocal_controls import load_ai_control_adapter, _resize_freq, _resize_time, _logit
from yuaz_ddsp_resampler.state import lookup_local_record, find_voicebank_for_input
from yuaz_ddsp_resampler.vocal_controls import apply_decoder_vocal_controls

root=Path(sys.argv[1]).resolve(); wav=Path(sys.argv[2]).expanduser().resolve()
record=lookup_local_record(wav) or {}
pack_path=Path(str(record.get('ai_phonation_adapter') or '')).expanduser()
if not pack_path.is_file(): raise RuntimeError(f'active phonation pack missing: {pack_path}')
pack,meta=load_ai_control_adapter(pack_path,device='cpu',expected_controls=('tension','voicing'))
feat=NativeYuazDDSPExtractor(root).features(wav)
S=torch.from_numpy(np.exp(feat['log_spec']).astype(np.float32)).unsqueeze(0)
AP=torch.from_numpy(np.asarray(feat['ap'],dtype=np.float32)).unsqueeze(0)
G=torch.from_numpy(np.asarray(feat['gate'],dtype=np.float32)).unsqueeze(0)
F0=torch.from_numpy(np.asarray(feat['f0'],dtype=np.float32)).unsqueeze(0)

def ctrls(v):
    z=torch.zeros((1,1,S.shape[-1]),dtype=S.dtype)
    return {'tension':z.clone(),'voicing':torch.full_like(z,float(v))}

def raw(v):
    with torch.inference_mode(): return tuple(x.detach().float() for x in pack.predict_residuals(S,AP,G,F0,ctrls(v)))

def routed_neural(v):
    neg=raw(-1.0); pos=raw(1.0)
    odd_ap=0.5*(pos[1]-neg[1])
    da=odd_ap if v>0 else -odd_ap
    gain=float(getattr(pack,'runtime_gain',2.0))
    da_full=_resize_freq(da*gain,AP.shape[1]); da_full=_resize_time(da_full,AP.shape[-1])
    out_ap=torch.sigmoid(_logit(AP)+1.55*torch.tanh(da_full))
    active=_resize_time((F0>1.0).to(AP.dtype),AP.shape[-1])>0.5
    out_ap=torch.where(active,out_ap.clamp(0.012,0.988),AP)
    return S.clone(),out_ap,G.clone(),(torch.zeros_like(pos[0]),da,torch.zeros_like(pos[2]))

def routed_final(v):
    ns,na,ng,nraw=routed_neural(v)
    with torch.inference_mode():
        fs,fa,fg=apply_decoder_vocal_controls(ns,na,ng,F0,ctrls(v),sample_rate=24000,learned_controls=pack.control_names)
    return (torch.log(fs.clamp(min=1e-7)/S.clamp(min=1e-7)).detach().float(),(fa-AP).detach().float(),(fg-G).detach().float()), nraw

def old_applied(v):
    with torch.inference_mode(): os,oa,og=pack.apply(S,AP,G,F0,ctrls(v))
    return (torch.log(os.clamp(min=1e-7)/S.clamp(min=1e-7)).detach().float(),(oa-AP).detach().float(),(og-G).detach().float())

def rms(x): return float(torch.sqrt(torch.mean(x*x)+1e-12))
def cmp(a,b):
    af=a.reshape(-1);bf=b.reshape(-1);den=float(torch.linalg.vector_norm(af)*torch.linalg.vector_norm(bf))
    cos=float(torch.dot(af,bf)/den) if den>1e-12 else 0.0
    ra=rms(a);rb=rms(b);m=max(1e-8,.5*(ra+rb))
    return {'rms_negative':ra,'rms_positive':rb,'cosine':cos,'difference_ratio':rms(a-b)/m,'sum_ratio':rms(a+b)/m}
def emit(kind,a,b):
    for scope,x,y in zip(('spectral','ap','gate'),a,b):
        row={'type':kind,'scope':scope};row.update(cmp(x,y));print(json.dumps(row,separators=(',',':')))

print(json.dumps({'type':'inventory','voicebank':str(find_voicebank_for_input(wav) or ''),'wav':str(wav),'pack':str(pack_path),'runtime_gain':float(getattr(pack,'runtime_gain',2.0)),'route':'old-pack YV odd-AP only + deterministic signed spectral/gate; YT unchanged'},separators=(',',':')))
oldn=old_applied(-1.0);oldp=old_applied(1.0)
emit('old_neural_applied_yv',oldn,oldp)
rn,nraw=routed_final(-1.0);rp,praw=routed_final(1.0)
emit('routed_neural_raw_yv',nraw,praw)
emit('routed_final_yv',rn,rp)
print(json.dumps({'type':'yt_guarantee','statement':'Runtime route is conditional on voicing; YT-only path uses the existing phonation pack unchanged.'},separators=(',',':')))
PY
