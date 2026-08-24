#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
./commands/run.command install-openutau-macos
RUNTIME="$HOME/Library/Application Support/YuazDDSP/0.2.9"
AI="$RUNTIME/src/yuaz_ddsp_resampler/ai_vocal_controls.py"
VOCAL="$RUNTIME/src/yuaz_ddsp_resampler/vocal_controls.py"
python3 - "$AI" "$VOCAL" <<'PY'
from pathlib import Path
import sys

ai = Path(sys.argv[1])
s = ai.read_text(encoding="utf-8")
old = '''        if float(torch.max(torch.abs(v)).detach().cpu()) <= 1e-6:\n            return torch.zeros_like(t_ds), torch.zeros_like(t_da), torch.zeros_like(t_dg)\n'''
new = '''        if float(torch.max(torch.abs(v)).detach().cpu()) <= 1e-6:\n            return t_ds, t_da, t_dg\n'''
if old not in s:
    raise SystemExit("phonation bypass patch point not found")
s = s.replace(old, new, 1)
old = '''        active_s = (activity > 0) & (f > 1.0)\n        active_ap = _resize_time(active_s.to(ap_bands.dtype), ap_bands.shape[-1]) > 0.5\n        active_gate = _resize_time(active_s.to(gate.dtype), gate.shape[-1]) > 0.5\n        out_s = torch.where(active_s, out_s.clamp(min=1e-7), spectral_envelope)\n        out_ap = torch.where(active_ap, out_ap.clamp(0.012, 0.988), ap_bands)\n        out_gate = torch.where(active_gate, out_gate.clamp(0.02, 0.98), gate)\n'''
new = '''        active_s = (activity > 0) & (f > 1.0)\n        spectral_effect = torch.amax(torch.abs(ds_full), dim=1, keepdim=True) > 1e-7\n        active_s = active_s & spectral_effect\n        active_ap = (_resize_time((activity > 0).to(ap_bands.dtype), ap_bands.shape[-1]) > 0.5)\n        active_ap = active_ap & (torch.amax(torch.abs(da_full), dim=1, keepdim=True) > 1e-7)\n        active_gate = (_resize_time((activity > 0).to(gate.dtype), gate.shape[-1]) > 0.5)\n        active_gate = active_gate & (torch.abs(dg_full) > 1e-7)\n        out_s = torch.where(active_s, out_s.clamp(min=1e-7), spectral_envelope)\n        out_ap = torch.where(active_ap, out_ap.clamp(0.012, 0.988), ap_bands)\n        out_gate = torch.where(active_gate, out_gate.clamp(0.02, 0.98), gate)\n'''
if old not in s:
    raise SystemExit("effect mask patch point not found")
s = s.replace(old, new, 1)
ai.write_text(s, encoding="utf-8")

vocal = Path(sys.argv[2])
s = vocal.read_text(encoding="utf-8")
replacements = [
    ('tension_scale = carrier("tension", 0.74)', 'tension_scale = carrier("tension", 0.88)'),
    ('tension_eff = torch.sign(tension) * torch.pow(torch.abs(tension), 1.20) * tension_scale',
     'tension_pos_eff = torch.pow(torch.clamp(tension, 0.0, 1.0), 1.05) * tension_scale\n    tension_neg_eff = torch.pow(torch.clamp(-tension, 0.0, 1.0), 0.95) * tension_scale * 1.08\n    tension_eff = tension_pos_eff - tension_neg_eff'),
    ('tension_shape = -0.26 * low + 0.20 * mid + 0.30 * upper',
     'tension_shape = -0.34 * low + 0.28 * mid + 0.42 * upper'),
    ('tension_gain = torch.exp(0.68 * tension_eff * tension_shape * voiced)',
     'tension_gain = torch.exp(0.86 * tension_eff * tension_shape * voiced)'),
    ('t_ap = torch.zeros_like(_interp_curve(tension_eff, ap_frames, ap_bands.device, ap_bands.dtype))',
     't_ap = _interp_curve(tension_eff, ap_frames, ap_bands.device, ap_bands.dtype)'),
    ('out_ap = out_ap - 0.16 * t_pos * tension_ap_shape * out_ap',
     'out_ap = out_ap - 0.10 * t_pos * tension_ap_shape * out_ap'),
    ('out_ap = out_ap + 0.18 * t_neg * tension_ap_shape * (1.0 - out_ap)',
     'out_ap = out_ap + 0.14 * t_neg * tension_ap_shape * (1.0 - out_ap)'),
    ('t_g = torch.zeros_like(_interp_curve(tension_eff, gate_frames, gate.device, gate.dtype))',
     't_g = _interp_curve(tension_eff, gate_frames, gate.device, gate.dtype)'),
    ('out_gate = out_gate + 0.08 * t_pos_g * (1.0 - out_gate)',
     'out_gate = out_gate + 0.05 * t_pos_g * (1.0 - out_gate)'),
    ('out_gate = out_gate - 0.10 * t_neg_g * out_gate',
     'out_gate = out_gate - 0.08 * t_neg_g * out_gate'),
]
for old, new in replacements:
    if old not in s:
        raise SystemExit("vocal patch point not found: " + old)
    s = s.replace(old, new, 1)
vocal.write_text(s, encoding="utf-8")
PY
pkill -f yuaz_ddsp_resampler.server 2>/dev/null || true
echo "Installed balanced signed YT test"
