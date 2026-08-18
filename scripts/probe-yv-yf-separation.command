#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || exit 1
WAV="${1:-}"
if [ -z "$WAV" ]; then
  read -r WAV
fi
[ -f "$WAV" ] || exit 1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$ROOT" "$WAV" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
import torch
from yuaz_ddsp_resampler.ai_control_training import NativeYuazDDSPExtractor
from yuaz_ddsp_resampler.ai_vocal_controls import load_ai_control_adapter
from yuaz_ddsp_resampler.state import lookup_local_record
from yuaz_ddsp_resampler.vocal_controls import apply_decoder_vocal_controls

root=Path(sys.argv[1]).resolve()
wav=Path(sys.argv[2]).expanduser().resolve()
record=lookup_local_record(wav) or {}
tech_path=Path(str(record.get('ai_control_adapter') or '')).expanduser()
phon_path=Path(str(record.get('ai_phonation_adapter') or '')).expanduser()
if not tech_path.is_file(): raise RuntimeError('technique pack missing')
if not phon_path.is_file(): raise RuntimeError('phonation pack missing')
tech,tmeta=load_ai_control_adapter(tech_path,device='cpu')
phon,pmeta=load_ai_control_adapter(phon_path,device='cpu')
feat=NativeYuazDDSPExtractor(root).features(wav)
S=torch.from_numpy(np.exp(feat['log_spec']).astype(np.float32)).unsqueeze(0)
AP=torch.from_numpy(np.asarray(feat['ap'],dtype=np.float32)).unsqueeze(0)
G=torch.from_numpy(np.asarray(feat['gate'],dtype=np.float32)).unsqueeze(0)
F0=torch.from_numpy(np.asarray(feat['f0'],dtype=np.float32)).unsqueeze(0)

def zeros(names):
    return {n:torch.zeros((1,1,S.shape[-1]),dtype=S.dtype) for n in names}

def raw(pack,name,value):
    c=zeros(pack.control_names); c[name].fill_(float(value))
    with torch.inference_mode():
        return tuple(x.detach().float() for x in pack.predict_residuals(S,AP,G,F0,c))

def applied(pack,name,value):
    c=zeros(pack.control_names); c[name].fill_(float(value))
    with torch.inference_mode():
        os,oa,og=pack.apply(S,AP,G,F0,c)
    ds=torch.log(os.clamp(min=1e-7)/S.clamp(min=1e-7))
    return ds.detach().float(),(oa-AP).detach().float(),(og-G).detach().float()

def carrier(name,value,learned):
    names=('tension','breathiness','voicing','gender_formant','mouth','falsetto','mixed_voice','pharyngeal')
    c={n:torch.zeros((1,1,S.shape[-1]),dtype=S.dtype) for n in names}
    c[name].fill_(float(value))
    with torch.inference_mode():
        os,oa,og=apply_decoder_vocal_controls(S,AP,G,F0,c,sample_rate=24000,learned_controls=learned)
    ds=torch.log(os.clamp(min=1e-7)/S.clamp(min=1e-7))
    return ds.detach().float(),(oa-AP).detach().float(),(og-G).detach().float()

def rms(x):
    return float(torch.sqrt(torch.mean(x*x)+1e-12))

def compare(a,b):
    af=a.reshape(-1); bf=b.reshape(-1)
    den=float(torch.linalg.vector_norm(af)*torch.linalg.vector_norm(bf))
    cos=float(torch.dot(af,bf)/den) if den>1e-12 else 0.0
    ra=rms(a); rb=rms(b); mean=max(1e-8,0.5*(ra+rb))
    return {'rms_a':ra,'rms_b':rb,'cosine':cos,'difference_ratio':rms(a-b)/mean,'sum_ratio':rms(a+b)/mean}

def emit(kind,pair,a,b):
    for scope,x,y in zip(('spectral','ap','gate'),a,b):
        row={'type':kind,'pair':pair,'scope':scope}
        row.update(compare(x,y))
        print(json.dumps(row,separators=(',',':')))

print(json.dumps({'type':'packs','technique':str(tech_path),'phonation':str(phon_path),'technique_controls':list(tech.control_names),'phonation_controls':list(phon.control_names)},separators=(',',':')))

yb_raw=raw(tech,'breathiness',1.0); yf_raw=raw(tech,'falsetto',1.0)
yb_app=applied(tech,'breathiness',1.0); yf_app=applied(tech,'falsetto',1.0)
yb_car=carrier('breathiness',1.0,tech.control_names); yf_car=carrier('falsetto',1.0,tech.control_names)
emit('yb_yf_raw','YB100_vs_YF100',yb_raw,yf_raw)
emit('yb_yf_applied','YB100_vs_YF100',yb_app,yf_app)
emit('yb_yf_carrier','YB100_vs_YF100',yb_car,yf_car)

yv_neg_raw=raw(phon,'voicing',-1.0); yv_pos_raw=raw(phon,'voicing',1.0)
yv_neg_app=applied(phon,'voicing',-1.0); yv_pos_app=applied(phon,'voicing',1.0)
yv_neg_car=carrier('voicing',-1.0,phon.control_names); yv_pos_car=carrier('voicing',1.0,phon.control_names)
emit('yv_raw','YV-100_vs_YV100',yv_neg_raw,yv_pos_raw)
emit('yv_applied','YV-100_vs_YV100',yv_neg_app,yv_pos_app)
emit('yv_carrier','YV-100_vs_YV100',yv_neg_car,yv_pos_car)
PY
