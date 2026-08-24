#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
./commands/run.command install-openutau-macos
RUNTIME="$HOME/Library/Application Support/YuazDDSP/0.2.9"
TARGET="$RUNTIME/src/yuaz_ddsp_resampler/ai_vocal_controls.py"
python3 - "$TARGET" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
old = '''        if float(torch.max(torch.abs(v)).detach().cpu()) <= 1e-6:\n            return torch.zeros_like(t_ds), torch.zeros_like(t_da), torch.zeros_like(t_dg)\n'''
new = '''        if float(torch.max(torch.abs(v)).detach().cpu()) <= 1e-6:\n            return t_ds, t_da, t_dg\n'''
if old not in s:
    raise SystemExit("phonation bypass patch point not found")
s = s.replace(old, new, 1)
old = '''        active_s = (activity > 0) & (f > 1.0)\n        active_ap = _resize_time(active_s.to(ap_bands.dtype), ap_bands.shape[-1]) > 0.5\n        active_gate = _resize_time(active_s.to(gate.dtype), gate.shape[-1]) > 0.5\n        out_s = torch.where(active_s, out_s.clamp(min=1e-7), spectral_envelope)\n        out_ap = torch.where(active_ap, out_ap.clamp(0.012, 0.988), ap_bands)\n        out_gate = torch.where(active_gate, out_gate.clamp(0.02, 0.98), gate)\n'''
new = '''        active_s = (activity > 0) & (f > 1.0)\n        spectral_effect = torch.amax(torch.abs(ds_full), dim=1, keepdim=True) > 1e-7\n        active_s = active_s & spectral_effect\n        active_ap = (_resize_time(active_s.to(ap_bands.dtype), ap_bands.shape[-1]) > 0.5)\n        active_ap = active_ap & (torch.amax(torch.abs(da_full), dim=1, keepdim=True) > 1e-7)\n        active_gate = (_resize_time(active_s.to(gate.dtype), gate.shape[-1]) > 0.5)\n        active_gate = active_gate & (torch.abs(dg_full) > 1e-7)\n        out_s = torch.where(active_s, out_s.clamp(min=1e-7), spectral_envelope)\n        out_ap = torch.where(active_ap, out_ap.clamp(0.012, 0.988), ap_bands)\n        out_gate = torch.where(active_gate, out_gate.clamp(0.02, 0.98), gate)\n'''
if old not in s:
    raise SystemExit("effect mask patch point not found")
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
PY
pkill -f yuaz_ddsp_resampler.server 2>/dev/null || true
echo "Installed YT continuous learned-residual test"
