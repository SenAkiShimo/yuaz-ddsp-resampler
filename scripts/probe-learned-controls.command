#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
"$ROOT/scripts/import-existing-control-packs.command" >/dev/null || true
[ -x "$ROOT/.venv/bin/python" ] || { echo "Run setup-macos.command first."; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$ROOT/.venv/bin/python" - <<'PY'
from pathlib import Path
import torch
from yuaz_ddsp_resampler.ai_vocal_controls import load_ai_control_adapter

root=Path.cwd(); cm=root/'control_models'
expected=[
 ('ai_control_foundation-v2.pt',('breathiness','falsetto','mixed_voice','pharyngeal')),
 ('ai_gender_foundation-v1.pt',('gender_formant',)),
 ('ai_phonation_foundation-v1.pt',('tension','voicing')),
 ('ai_mouth_foundation-v1.pt',('mouth',)),
]
T=120
freq=torch.linspace(0,1,64).view(1,64,1)
time=torch.linspace(0,6.283185,T).view(1,1,T)
S=torch.exp(-1.15*freq)*(1.0+0.18*torch.sin(time)+0.12*torch.cos(freq*16.0)).clamp(min=.2)
S=S.expand(1,64,T).contiguous()
apf=torch.linspace(.08,.32,16).view(1,16,1)
AP=(apf*(1.0+0.08*torch.sin(time))).expand(1,16,T).contiguous().clamp(.02,.8)
G=torch.full((1,1,T),.72)
F0=torch.full((1,1,T),220.)
print('Yuaz 0.2.8ai.13 learned-control effect probe')
all_ok=True
for filename,names in expected:
 p=cm/filename
 if not p.is_file():
  print(f'MISSING {filename}')
  all_ok=False; continue
 pack,meta=load_ai_control_adapter(p,device='cpu',expected_controls=names)
 print(f'\n[{filename}] controls={pack.control_names} scopes={pack.output_scopes} base_gain={pack.runtime_gain:.2f}')
 for i,(name,mode) in enumerate(zip(pack.control_names,pack.control_modes)):
  vals=(1.0,) if mode=='positive' else (-1.0,1.0)
  for value in vals:
   controls={n:torch.zeros(1,1,T) for n in pack.control_names}
   controls[name]=torch.full((1,1,T),value)
   with torch.inference_mode():
    pack.apply(S,AP,G,F0,controls)
   st=pack.last_effect_stats
   applied=max(st.get('applied_spectral_log_rms',0),st.get('applied_ap_rms',0),st.get('applied_gate_rms',0))
   raw=max(st.get('raw_spectral_rms',0),st.get('raw_ap_rms',0),st.get('raw_gate_rms',0))
   status='PASS' if applied>=1e-4 else 'COLLAPSED'
   if status!='PASS': all_ok=False
   sign='+' if value>0 else '-'
   print(f'  {name:14s} {sign}100 raw={raw:.6f} applied={applied:.6f} gain={st.get("runtime_gain",1):.2f} {status}')
print('\nRESULT:', 'learned residuals are non-zero' if all_ok else 'one or more packs are missing/collapsed; 0.2.8ai.13 carrier fallback remains active')
PY
