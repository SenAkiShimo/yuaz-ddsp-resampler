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
import torch
import torch.nn.functional as F
from yuaz_ddsp_resampler.ai_control_training import NativeYuazDDSPExtractor
from yuaz_ddsp_resampler.ai_vocal_controls import load_ai_control_adapter
from yuaz_ddsp_resampler.state import find_voicebank_for_input,resolve_active_state
root=Path(sys.argv[1]).resolve()
wav=Path(sys.argv[2]).expanduser().resolve()
bank=find_voicebank_for_input(wav)
if bank is None:
    raise RuntimeError('voicebank not found')
state,_=resolve_active_state(bank,verify=True)
if state is None:
    raise RuntimeError('active ai14 state not found')
pack,meta=load_ai_control_adapter(state/'ai_phonation_adapter.ai14.pt',device='cpu',expected_controls=('tension','voicing'))
native=NativeYuazDDSPExtractor(root)
feat=native.features(wav)
T=min(feat['log_spec'].shape[-1],max(1,int(round(native.sample_rate/native.hop))))
S=torch.from_numpy(feat['log_spec'][:,:T]).unsqueeze(0).exp().float()
AP=torch.from_numpy(feat['ap'][:,:T]).unsqueeze(0).float()
G=torch.from_numpy(feat['gate'][:,:T]).unsqueeze(0).float()
F0=torch.full((1,1,T),261.625565)
score=G.clamp(0,1)*(1-AP.mean(dim=1,keepdim=True).clamp(0,1))
print(json.dumps({
    'frames':T,
    'gate_mean':float(G.mean()),
    'ap_mean':float(AP.mean()),
    'periodic_score_mean':float(score.mean()),
    'periodic_score_max':float(score.max()),
    'periodic_fraction_gt_010':float((score>0.10).float().mean()),
},separators=(',',':')))
for value in (-1.0,1.0):
    controls={'tension':torch.full((1,1,T),value),'voicing':torch.zeros((1,1,T))}
    with torch.inference_mode():
        x,c,mask=pack._context(S,AP,G,F0,controls)
        h=F.silu(pack.input_proj(x))
        h=h+0.45*F.silu(pack.temporal1(h))
        h=h+0.45*F.silu(pack.temporal2(h))
        h=h+0.30*F.silu(pack.temporal3(h))
        y=pack.output_proj(h)
        pre_s=y[:,:pack.spectral_bands]
        pre_ap=y[:,pack.spectral_bands:pack.spectral_bands+pack.ap_bands]
        pre_gate=y[:,-1:]
        ds,da,dg=pack.predict_residuals(S,AP,G,F0,controls)
    def rms(z):
        return float(torch.sqrt(torch.mean(torch.tanh(z).pow(2))+1e-12))
    print(json.dumps({
        'yt':int(value*100),
        'mask_fraction':float(mask.mean()),
        'pre_s':rms(pre_s),
        'pre_ap':rms(pre_ap),
        'pre_gate':rms(pre_gate),
        'post_s':rms(ds),
        'post_ap':rms(da),
        'post_gate':rms(dg),
    },separators=(',',':')))
PY