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
from yuaz_ddsp_resampler.vocal_controls import apply_decoder_vocal_controls

root=Path(sys.argv[1]).resolve(); wav=Path(sys.argv[2]).expanduser().resolve()
record=lookup_local_record(wav) or {}
old_path=Path(str(record.get('ai_phonation_adapter') or '')).expanduser()
config=json.loads((root/'config.json').read_text(encoding='utf-8'))
sha=checkpoint_identity_sha(Path(config['checkpoint']).expanduser())
candidate=root/'control_models'/f'ai_phonation_foundation-v1-signedfix-{sha[:16]}.pt'
if not old_path.is_file(): raise RuntimeError(f'active phonation pack missing: {old_path}')
old,old_meta=load_ai_control_adapter(old_path,device='cpu',expected_controls=('tension','voicing'))
new=new_meta=None
if candidate.is_file():
    new,new_meta=load_ai_control_adapter(candidate,device='cpu',expected_controls=('tension','voicing'))
for label,meta in [('old',old_meta),('new',new_meta)]:
    if meta is not None and str(meta.get('checkpoint_sha256') or '')!=sha:
        raise RuntimeError(f'{label} checkpoint mismatch')

feat=NativeYuazDDSPExtractor(root).features(wav)
S=torch.from_numpy(np.exp(feat['log_spec']).astype(np.float32)).unsqueeze(0)
AP=torch.from_numpy(np.asarray(feat['ap'],dtype=np.float32)).unsqueeze(0)
G=torch.from_numpy(np.asarray(feat['gate'],dtype=np.float32)).unsqueeze(0)
F0=torch.from_numpy(np.asarray(feat['f0'],dtype=np.float32)).unsqueeze(0)

def controls(name,value):
    frames=S.shape[-1]
    out={n:torch.zeros((1,1,frames),dtype=S.dtype) for n in ('tension','breathiness','voicing','gender_formant','mouth','falsetto','mixed_voice','pharyngeal')}
    out[name].fill_(float(value))
    return out

def pack_controls(pack,all_controls):
    return {n:all_controls[n] for n in pack.control_names}

def routed(pack,name,value):
    c=controls(name,value)
    pc=pack_controls(pack,c)
    with torch.inference_mode():
        r=pack._phonation_routed_residuals(S,AP,G,F0,pc)
    if r is None:
        with torch.inference_mode():
            r=pack.predict_residuals(S,AP,G,F0,pc)
    return tuple(x.detach().float() for x in r)

def final_chain(pack,name,value):
    c=controls(name,value)
    pc=pack_controls(pack,c)
    with torch.inference_mode():
        ns,na,ng=pack.apply(S,AP,G,F0,pc)
        fs,fa,fg=apply_decoder_vocal_controls(ns,na,ng,F0,c,sample_rate=24000,learned_controls=pack.control_names)
    ds=torch.log(fs.clamp(min=1e-7)/S.clamp(min=1e-7))
    return ds.detach().float(),(fa-AP).detach().float(),(fg-G).detach().float()

def rms(x): return float(torch.sqrt(torch.mean(x*x)+1e-12))
def compare(a,b):
    af=a.reshape(-1); bf=b.reshape(-1)
    den=float(torch.linalg.vector_norm(af)*torch.linalg.vector_norm(bf))
    cos=float(torch.dot(af,bf)/den) if den>1e-12 else 0.0
    ra=rms(a); rb=rms(b); mean=max(1e-8,.5*(ra+rb))
    return {'rms_a':ra,'rms_b':rb,'cosine':cos,'difference_ratio':rms(a-b)/mean,'sum_ratio':rms(a+b)/mean}
def emit(kind,pack_label,control,a,b):
    for scope,x,y in zip(('spectral','ap','gate'),a,b):
        row={'type':kind,'pack':pack_label,'control':control,'scope':scope}; row.update(compare(x,y)); print(json.dumps(row,separators=(',',':')))

print(json.dumps({'type':'inventory','wav':str(wav),'voicebank':str(find_voicebank_for_input(wav) or ''),'checkpoint_sha256':sha,'old_pack':str(old_path),'candidate':str(candidate) if candidate.is_file() else '', 'runtime_route_expected':'phonation-yv-odd-ap-v2'},separators=(',',':')))

for label,pack in [('old',old),('candidate',new)]:
    if pack is None: continue
    rn=routed(pack,'voicing',-1.0); rp=routed(pack,'voicing',1.0)
    emit('routed_yv',label,'YV-100_vs_YV100',rn,rp)
    fn=final_chain(pack,'voicing',-1.0); fp=final_chain(pack,'voicing',1.0)
    emit('final_yv',label,'YV-100_vs_YV100',fn,fp)
    tn=final_chain(pack,'tension',-1.0); tp=final_chain(pack,'tension',1.0)
    emit('final_yt',label,'YT-100_vs_YT100',tn,tp)
PY
