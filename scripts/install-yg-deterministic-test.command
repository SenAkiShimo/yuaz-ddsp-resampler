#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
./commands/run.command install-openutau-macos
RUNTIME="$HOME/Library/Application Support/YuazDDSP/0.2.9"
VOCAL="$RUNTIME/src/yuaz_ddsp_resampler/vocal_controls.py"
AI="$RUNTIME/src/yuaz_ddsp_resampler/ai_vocal_controls.py"
python3 - "$VOCAL" "$AI" <<'PY'
from pathlib import Path
import sys

vocal = Path(sys.argv[1])
s = vocal.read_text(encoding="utf-8")
repl = [
    ('gender_scale = carrier("gender_formant", 0.85)', 'gender_scale = carrier("gender_formant", 1.0)'),
    ('gender_curve = torch.sign(gender_eff) * torch.pow(torch.abs(gender_eff), 0.86)',
     'gender_curve = torch.sign(gender_eff) * torch.pow(torch.abs(gender_eff), 0.78)'),
    ('formant_shift = 5.2 * gender_curve', 'formant_shift = 8.0 * gender_curve'),
    ('    gender_shape = (\n        0.22 * torch.exp(-0.5 * torch.square((hz - 900.0) / 950.0))\n        - 0.12 * torch.exp(-0.5 * torch.square((hz - 3150.0) / 1900.0))\n    )\n    gender_gain = torch.exp(0.22 * gender_eff * gender_shape * voiced)\n',
     '    gender_gain = torch.ones_like(out_s)\n'),
]
for old, new in repl:
    if old not in s:
        raise SystemExit("YG vocal patch point not found: " + old.split("\n", 1)[0])
    s = s.replace(old, new, 1)
vocal.write_text(s, encoding="utf-8")

ai = Path(sys.argv[2])
s = ai.read_text(encoding="utf-8")
needle = '''    def apply(self, spectral_envelope, ap_bands, gate, f0, controls):\n        routed = self._phonation_routed_residuals(\n'''
replacement = '''    def apply(self, spectral_envelope, ap_bands, gate, f0, controls):\n        if tuple(self.control_names) == ("gender_formant",):\n            self.last_effect_stats = {\n                "controls": {"gender_formant": float(torch.max(torch.abs(_curve(controls.get("gender_formant"), spectral_envelope.shape[-1], spectral_envelope.device, spectral_envelope.dtype))).detach().cpu())},\n                "runtime_route": "gender-deterministic-only-test",\n                "runtime_gain": 0.0,\n            }\n            return spectral_envelope, ap_bands, gate\n        routed = self._phonation_routed_residuals(\n'''
if needle not in s:
    raise SystemExit("YG AI bypass patch point not found")
s = s.replace(needle, replacement, 1)
ai.write_text(s, encoding="utf-8")
PY
pkill -f yuaz_ddsp_resampler.server 2>/dev/null || true
echo "Installed deterministic-only YG test"
