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
"$PY" - "$WAV" <<'PY'
import json,sys
from pathlib import Path
import torch
from yuaz_ddsp_resampler.ai_vocal_controls import load_ai_control_adapter
from yuaz_ddsp_resampler.state import find_voicebank_for_input,resolve_active_state
wav=Path(sys.argv[1]).expanduser().resolve()
bank=find_voicebank_for_input(wav)
if bank is None:
    raise RuntimeError('voicebank not found')
state,_=resolve_active_state(bank,verify=True)
if state is None:
    raise RuntimeError('active ai14 state not found')
p=state/'ai_phonation_adapter.ai14.pt'
pack,meta=load_ai_control_adapter(p,device='cpu',expected_controls=('tension','voicing'))
T=120
freq=torch.linspace(0,1,64).view(1,64,1)
time=torch.linspace(0,6.283185,T).view(1,1,T)
S=torch.exp(-1.15*freq)*(1.0+0.18*torch.sin(time)+0.12*torch.cos(freq*16.0)).clamp(min=.2)
S=S.expand(1,64,T).contiguous()
apf=torch.linspace(.08,.32,16).view(1,16,1)
AP=(apf*(1.0+0.08*torch.sin(time))).expand(1,16,T).contiguous().clamp(.02,.8)
G=torch.full((1,1,T),.72)
F0=torch.full((1,1,T),220.)
print(json.dumps({
    'state':state.name,
    'checkpoint':meta.get('checkpoint_sha256'),
    'output_weight_norm':float(pack.output_proj.weight.detach().norm()),
    'output_bias_norm':float(pack.output_proj.bias.detach().norm()),
},separators=(',',':')))
for value in (-1.0,1.0):
    controls={'tension':torch.full((1,1,T),value),'voicing':torch.zeros((1,1,T))}
    with torch.inference_mode():
        ds,da,dg=pack.predict_residuals(S,AP,G,F0,controls)
        pack.apply(S,AP,G,F0,controls)
    st=pack.last_effect_stats
    print(json.dumps({
        'yt':int(value*100),
        'direct_s':float(torch.sqrt(torch.mean(torch.tanh(ds).pow(2))+1e-12)),
        'direct_ap':float(torch.sqrt(torch.mean(torch.tanh(da).pow(2))+1e-12)),
        'direct_gate':float(torch.sqrt(torch.mean(torch.tanh(dg).pow(2))+1e-12)),
        'applied_s':st.get('applied_spectral_log_rms'),
        'applied_ap':st.get('applied_ap_rms'),
        'applied_gate':st.get('applied_gate_rms'),
        'gain':st.get('runtime_gain'),
        'collapsed':st.get('collapsed'),
    },separators=(',',':')))
PY