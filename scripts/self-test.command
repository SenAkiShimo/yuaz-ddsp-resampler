#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${ROOT}/.venv/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$ROOT" <<'PY'
import sys
from pathlib import Path
root=Path(sys.argv[1])
assert (root/'VERSION').read_text().strip()=='0.2.8ai.16'
client=(root/'src/yuaz_ddsp_resampler/client.py').read_text()
server=(root/'src/yuaz_ddsp_resampler/server.py').read_text()
assert 'ENGINE_VERSION = "0.2.8ai.16"' in client
assert 'DEFAULT_PORT = 47888' in client
assert 'ENGINE_VERSION = "0.2.8ai.16"' in server
controls=(root/'src/yuaz_ddsp_resampler/ai_vocal_controls.py').read_text()
assert 'mask = (strength > 1e-6).to(c.dtype) * voiced' in controls
assert 'control_gate_mode": "periodic-active"' in controls
assert 'mask = strength * voiced' not in controls
vocal=(root/'src/yuaz_ddsp_resampler/vocal_controls.py').read_text()
for expected in (
    'tension_scale = carrier("tension", 0.92)',
    'gender_scale = carrier("gender_formant", 0.65)',
    'mouth_scale = carrier("mouth", 0.95)',
    'mixed_scale = carrier("mixed_voice", 0.95, 0.95)',
    'pharyngeal_scale = carrier("pharyngeal", 0.95, 0.95)',
    'source_periodic = _periodic_mask(ap_bands, gate, frames, device, dtype)',
    'out_gate = out_gate + 0.62 * t_pos_g * (1.0 - out_gate)',
    'out_ap = out_ap - 0.58 * t_pos * tension_ap_shape * out_ap',
    'out_gate = out_gate + 0.48 * x_g * (1.0 - out_gate)',
    'shift_hz = (430.0 * f1_weight + 105.0 * f2_weight) * control',
):
    assert expected in vocal
install=(root/'scripts/install-openutau-macos.command').read_text()
assert '0.2.8ai.16' in install
assert '.yuaz-0.2.8ai14' in install
assert 'preserve_ai14' in install
purge=(root/'scripts/purge-previous-version.command').read_text()
assert '.yuaz-0.2.8ai13' in purge
assert '.yuaz-0.2.8ai14' in purge
prepare=(root/'scripts/prepare-voicebank.command').read_text()
deep=(root/'scripts/deep-train-voicebank.command').read_text()
assert 'disabled' in prepare.lower()
assert 'disabled' in deep.lower()
PY
python3 -m compileall -q "$ROOT/src"
while IFS= read -r -d '' f; do bash -n "$f"; done < <(find "$ROOT" -type f -name '*.command' -not -path '*/previous_versions/*' -print0)
echo "0.2.8ai.16 self-test OK"
