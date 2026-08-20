#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${ROOT}/.venv/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$ROOT" <<'PY'
import sys
from pathlib import Path
root=Path(sys.argv[1])

def require(condition, label):
    if not condition:
        raise AssertionError(f"self-test failed: {label}")

require((root/'VERSION').read_text().strip()=='0.2.9', 'VERSION is not 0.2.9')
client=(root/'src/yuaz_ddsp_resampler/client.py').read_text()
server=(root/'src/yuaz_ddsp_resampler/server.py').read_text()
require('ENGINE_VERSION = "0.2.9"' in client, 'client engine version')
require('DEFAULT_PORT = 47888' in client, 'client production port')
require('ENGINE_VERSION = "0.2.9"' in server, 'server engine version')
controls=(root/'src/yuaz_ddsp_resampler/ai_vocal_controls.py').read_text()
require('mask = (strength > 1e-6).to(c.dtype) * voiced' in controls, 'AI control voiced mask')
require('control_gate_mode": "source-active-voiced"' in controls, 'AI control gate mode')
require('mask = strength * voiced' not in controls, 'obsolete strength mask still present')
vocal=(root/'src/yuaz_ddsp_resampler/vocal_controls.py').read_text()
for label, expected in (
    ('YT carrier', 'tension_scale = carrier("tension", 0.92)'),
    ('YG carrier', 'gender_scale = carrier("gender_formant", 0.65)'),
    ('YO carrier', 'mouth_scale = carrier("mouth", 0.95)'),
    ('YF carrier', 'falsetto_spectral_scale = carrier("falsetto", 0.88, 0.96)'),
    ('YX carrier', 'mixed_scale = carrier("mixed_voice", 0.95, 0.95)'),
    ('YP carrier', 'pharyngeal_scale = carrier("pharyngeal", 0.95, 0.95)'),
    ('YT AP route', 'out_ap = out_ap - 0.58 * t_pos * tension_ap_shape * out_ap'),
    ('YT gate route', 'out_gate = out_gate + 0.62 * t_pos_g * (1.0 - out_gate)'),
    ('YX gate route', 'out_gate = out_gate + 0.48 * x_g * (1.0 - out_gate)'),
    ('YO formant warp', 'shift_hz = (430.0 * f1_weight + 105.0 * f2_weight) * control'),
    ('YF F0-relative register', 'harmonic_order = hz / f0_env'),
):
    require(expected in vocal, label)
install=(root/'scripts/install-openutau-macos.command').read_text()
require('0.2.9' in install, 'installer version')
require('.yuaz-0.2.8ai14' in install, 'ai.14 state namespace')
require('preserve_ai14' in install, 'ai.14 preservation')
prepare=(root/'scripts/prepare-voicebank.command').read_text()
deep=(root/'scripts/deep-train-voicebank.command').read_text()
require('disabled' in prepare.lower(), 'voicebank preparation must remain disabled')
require('disabled' in deep.lower(), 'voicebank deep training must remain disabled')
PY
python3 -m compileall -q "$ROOT/src"
while IFS= read -r -d '' f; do bash -n "$f"; done < <(find "$ROOT" -type f -name '*.command' -not -path '*/previous_versions/*' -print0)
echo "0.2.9 self-test OK"
