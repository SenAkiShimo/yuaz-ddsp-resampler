#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${ROOT}/.venv/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$ROOT" <<'PY'
import sys
from pathlib import Path
root=Path(sys.argv[1])
assert (root/'VERSION').read_text().strip()=='0.2.8ai.15', 'VERSION is not 0.2.8ai.15'

client=(root/'src/yuaz_ddsp_resampler/client.py').read_text()
server=(root/'src/yuaz_ddsp_resampler/server.py').read_text()
assert 'ENGINE_VERSION = "0.2.8ai.15"' in client, 'client version mismatch'
assert 'DEFAULT_PORT = 47887' in client, 'ai.15 client port mismatch'
assert 'ENGINE_VERSION = "0.2.8ai.15"' in server, 'server version mismatch'

controls=(root/'src/yuaz_ddsp_resampler/ai_vocal_controls.py').read_text()
assert 'mask = (strength > 1e-6).to(c.dtype) * voiced' in controls, 'single-scaling learned-control gate missing'
assert 'control_gate_mode' in controls, 'control gate diagnostic missing'
assert 'mask = strength * voiced' not in controls, 'old double-scaling path still present'

vocal=(root/'src/yuaz_ddsp_resampler/vocal_controls.py').read_text()
for expected in (
    'tension_scale = carrier("tension", 0.70)',
    'gender_scale = carrier("gender_formant", 0.65)',
    'mouth_scale = carrier("mouth", 0.65)',
    'mixed_scale = carrier("mixed_voice", 0.62, 0.62)',
    'pharyngeal_scale = carrier("pharyngeal", 0.62, 0.62)',
    'out_gate = out_gate + 0.24 * t_pos_g * (1.0 - out_gate)',
    'out_ap = out_ap - 0.30 * t_pos * tension_ap_shape * out_ap',
):
    assert expected in vocal, f'missing control-calibration marker: {expected}'

install=(root/'scripts/install-openutau-macos.command').read_text()
assert '0.2.8ai.15' in install, 'ai.15 install identity missing'
assert '.yuaz-0.2.8ai14' in install, 'ai.14 state preservation marker missing from installer'
assert 'preserve_ai14' in install, 'preserve_ai14 config assertion missing from installer'
assert 'purge-previous-version.command' in install, 'ai.13 purge hook missing from installer'

purge=(root/'scripts/purge-previous-version.command').read_text()
assert '.yuaz-0.2.8ai13' in purge, 'ai.13 state cleanup missing'
assert '.yuaz-0.2.8ai14' in purge, 'ai.14 preservation guard missing'
assert 'PRESERVED' in purge, 'ai.14 preservation report missing'

prepare=(root/'scripts/prepare-voicebank.command').read_text()
deep=(root/'scripts/deep-train-voicebank.command').read_text()
assert 'disabled' in prepare.lower(), 'ai.15 Prepare must remain disabled in calibration build'
assert 'disabled' in deep.lower(), 'ai.15 Deep must remain disabled in calibration build'

print('ai.15 learned-residual single-scaling: OK')
print('ai.15 deterministic carrier floor: OK')
print('ai.15 tension AP/gate prior: OK')
print('ai.14 state preservation policy: OK')
print('ai.13 purge policy: OK')
PY
python3 -m compileall -q "$ROOT/src"
while IFS= read -r -d '' f; do bash -n "$f"; done < <(find "$ROOT" -type f -name '*.command' -not -path '*/previous_versions/*' -print0)
echo "0.2.8ai.15 control-calibration self-test OK"
