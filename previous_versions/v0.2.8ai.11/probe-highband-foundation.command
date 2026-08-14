#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.venv/bin/python"
MODEL="$ROOT/control_models/highband_foundation-v2.pt"
[ -f "$MODEL" ] || MODEL="$ROOT/control_models/highband_foundation-v1.pt"
[ -x "$PY" ] || { echo "Run ./setup-macos.command first."; exit 1; }
[ -f "$MODEL" ] || { echo "Missing highband_foundation-v2.pt/v1.pt"; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$MODEL" <<'PY'
import json,sys,time
import torch
from yuaz_ddsp_resampler.highband_foundation import inspect_highband_foundation,load_highband_foundation
p=sys.argv[1]
print(json.dumps(inspect_highband_foundation(p),ensure_ascii=False,indent=2))
m,_=load_highband_foundation(p,device='cpu')
x=torch.randn(1,1,48000)*0.05
f=torch.full((1,1,1),110.0)
with torch.inference_mode():
    for _ in range(2): m(x,f)
    times=[]
    for _ in range(5):
        t=time.perf_counter(); m(x,f); times.append(time.perf_counter()-t)
median=sorted(times)[len(times)//2]
print(f"1.0 s CPU model-only median: {median:.4f} s  | model-only RTF {median:.4f}")
PY
