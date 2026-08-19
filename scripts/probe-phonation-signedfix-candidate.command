#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "Python not found: $PY" >&2; exit 1; }
INPUT="${1:-}"
if [ -z "$INPUT" ]; then
  read -r INPUT
fi
INPUT="${INPUT%/}"
WAV="$INPUT"
if [ -d "$INPUT" ]; then
  WAV="$(find "$INPUT" -type f -name 'additional.wav' -print | LC_ALL=C sort | head -n 1)"
  if [ -z "$WAV" ]; then
    WAV="$(find "$INPUT" -type f \( -iname '*.wav' -o -iname '*.wave' \) -print | LC_ALL=C sort | head -n 1)"
  fi
  [ -n "$WAV" ] || { echo "No WAV found under voicebank: $INPUT" >&2; exit 1; }
  echo "Selected WAV: $WAV"
elif [ ! -f "$WAV" ]; then
  echo "WAV or voicebank folder not found: $INPUT" >&2
  exit 1
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$ROOT" "$WAV" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
import torch
from yuaz_ddsp_resampler.ai_control_training import NativeYuazDDSPExtractor
from yuaz_ddsp_resampler.ai_vocal_controls import load_ai_control_adapter
from yuaz_ddsp_resampler.checkpoint_identity import checkpoint_identity_sha
from yuaz_ddsp_resampler.state import find_voicebank_for_input, lookup_local_record

root=Path(sys.argv[1]).resolve(); wav=Path(sys.argv[2]).expanduser().resolve()
record=lookup_local_record(wav) or {}
old_path=Path(str(record.get('ai_phonation_adapter') or '')).expanduser()
config=json.loads((root/'config.json').read_text(encoding='utf-8'))
sha=checkpoint_identity_sha(Path(config['checkpoint']).expanduser())
candidate=root/'control_models'/f'ai_phonation_foundation-v1-signedfix-{sha[:16]}.pt'
if not old_path.is_file(): raise RuntimeError(f'active phonation pack missing: {old_path}')
if not candidate.is_file(): raise RuntimeError(f'signedfix candidate missing: {candidate}')
old,old_meta=load_ai_control_adapter(old_path,device='cpu',expected_controls=('tension','voicing'))
new,new_meta=load_ai_control_adapter(candidate,device='cpu',expected_controls=('tension','voicing'))
for label,meta in [('old',old_meta),('new',new_meta)]:
    if str(meta.get('checkpoint_sha256') or '')!=sha: raise RuntimeError(f'{label} checkpoint mismatch')
feat=NativeYuazDDSPExtractor(root).features(wav)
S=torch.from_numpy(np.exp(feat['log_spec']).astype(np.float32)).unsqueeze(0)
AP=torch.from_numpy(np.asarray(feat['ap'],dtype=np.float32)).unsqueeze(0)
G=torch.from_numpy(np.asarray(feat['gate'],dtype=np.float32)).unsqueeze(0)
F0=torch.from_numpy(np.asarray(feat['f0'],dtype=np.float32)).unsqueeze(0)

def run(pack,name,value):
    controls={n:torch.zeros((1,1,S.shape[-1]),dtype=S.dtype) for n in pack.control_names}
    controls[name].fill_(float(value))
    with torch.inference_mode():
        raw=tuple(x.detach().float() for x in pack.predict_residuals(S,AP,G,F0,controls))
        os,oa,og=pack.apply(S,AP,G,F0,controls)
    applied=(torch.log(os.clamp(min=1e-7)/S.clamp(min=1e-7)).detach().float(),(oa-AP).detach().float(),(og-G).detach().float())
    return raw,applied

def rms(x): return float(torch.sqrt(torch.mean(x*x)+1e-12))
def compare(a,b):
    af=a.reshape(-1); bf=b.reshape(-1); den=float(torch.linalg.vector_norm(af)*torch.linalg.vector_norm(bf))
    cos=float(torch.dot(af,bf)/den) if den>1e-12 else 0.0
    ra=rms(a); rb=rms(b); mean=max(1e-8,.5*(ra+rb))
    return {'rms_a':ra,'rms_b':rb,'cosine':cos,'difference_ratio':rms(a-b)/mean,'sum_ratio':rms(a+b)/mean}
def emit(kind,pair,a,b):
    for scope,x,y in zip(('spectral','ap','gate'),a,b):
        row={'type':kind,'pair':pair,'scope':scope}; row.update(compare(x,y)); print(json.dumps(row,separators=(',',':')))

def magnitude(pack,name,value):
    raw,app=run(pack,name,value)
    return {'raw':{k:rms(v) for k,v in zip(('spectral','ap','gate'),raw)},'applied':{k:rms(v) for k,v in zip(('spectral','ap','gate'),app)}}

print(json.dumps({'type':'inventory','wav':str(wav),'voicebank':str(find_voicebank_for_input(wav) or ''),'old_pack':str(old_path),'candidate':str(candidate),'old_training_method':old_meta.get('training_method'),'candidate_training_method':new_meta.get('training_method'),'checkpoint_sha256':sha},separators=(',',':')))

for pack_label,pack in [('old',old),('new',new)]:
    nraw,napp=run(pack,'voicing',-1.0); praw,papp=run(pack,'voicing',1.0)
    emit(f'{pack_label}_yv_raw','YV-100_vs_YV100',nraw,praw)
    emit(f'{pack_label}_yv_applied','YV-100_vs_YV100',napp,papp)

for value in (-1.0,1.0):
    old_raw,old_app=run(old,'tension',value); new_raw,new_app=run(new,'tension',value)
    emit('yt_old_vs_new_raw',f'YT{int(value*100):+d}',old_raw,new_raw)
    emit('yt_old_vs_new_applied',f'YT{int(value*100):+d}',old_app,new_app)

print(json.dumps({'type':'yt_magnitudes','old_neg':magnitude(old,'tension',-1.0),'new_neg':magnitude(new,'tension',-1.0),'old_pos':magnitude(old,'tension',1.0),'new_pos':magnitude(new,'tension',1.0)},separators=(',',':')))
PY
