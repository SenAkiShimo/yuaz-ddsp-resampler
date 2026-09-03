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

require((root/'VERSION').read_text().strip()=='0.3.0', 'VERSION is not 0.3.0')
client=(root/'src/yuaz_ddsp_resampler/client.py').read_text()
server=(root/'src/yuaz_ddsp_resampler/server.py').read_text()
require('ENGINE_VERSION = "0.3.0"' in client, 'client engine version')
require('DEFAULT_PORT = 47889' in client, 'client production port')
require('ENGINE_VERSION = "0.3.0"' in server, 'server engine version')
controls=(root/'src/yuaz_ddsp_resampler/ai_vocal_controls.py').read_text()
require('mask = (strength > 1e-6).to(c.dtype) * voiced' in controls, 'AI control voiced mask')
require('control_gate_mode": "source-active-voiced"' in controls, 'AI control gate mode')
require('mask = strength * voiced' not in controls, 'obsolete strength mask still present')
require('t_progress = t_amount * t_amount * (3.0 - 2.0 * t_amount)' in controls, 'YT learned monotonic progress')
require('t_dg = t_dg * t_progress * (0.015 + 0.035 * t_progress)' in controls, 'YT learned gate reduction')
require('return t_ds, t_da, t_dg' in controls, 'YT learned residual continuity')
require('spectral_effect = torch.amax(torch.abs(ds_full), dim=1, keepdim=True) > 1e-7' in controls, 'AI zero-effect mask')
vocal=(root/'src/yuaz_ddsp_resampler/vocal_controls.py').read_text()
for label, expected in (
    ('YT carrier', 'tension_scale = carrier("tension", 0.88)'),
    ('YG carrier', 'gender_scale = carrier("gender_formant", 0.85)'),
    ('YO carrier', 'mouth_scale = carrier("mouth", 0.95)'),
    ('YF carrier', 'falsetto_spectral_scale = carrier("falsetto", 0.88, 0.96)'),
    ('YX carrier', 'mixed_scale = carrier("mixed_voice", 0.95, 0.95)'),
    ('YP carrier', 'pharyngeal_scale = carrier("pharyngeal", 0.95, 0.95)'),
    ('YT positive curve', 'torch.pow(torch.clamp(tension, 0.0, 1.0), 1.05) * tension_scale'),
    ('YT negative curve', 'torch.pow(torch.clamp(-tension, 0.0, 1.0), 0.95) * tension_scale * 1.08'),
    ('YT AP route', 'out_ap = out_ap - 0.10 * t_pos * tension_ap_shape * out_ap'),
    ('YT gate route', 'out_gate = out_gate + 0.05 * t_pos_g * (1.0 - out_gate)'),
    ('YX gate route', 'out_gate = out_gate + 0.48 * x_g * (1.0 - out_gate)'),
    ('YO formant warp', 'shift_hz = (430.0 * f1_weight + 105.0 * f2_weight) * control'),
    ('YF F0-relative register', 'harmonic_order = hz / f0_env'),
):
    require(expected in vocal, label)
install=(root/'scripts/install-openutau-macos.command').read_text()
require('0.3.0' in install, 'installer version')
require('.yuaz-0.2.8ai14' in install, 'ai.14 state namespace')
require('preserve_ai14' in install, 'ai.14 preservation')
prepare=(root/'scripts/prepare-voicebank.command').read_text()
deep=(root/'scripts/deep-train-voicebank.command').read_text()
require('disabled' in prepare.lower(), 'voicebank preparation must remain disabled')
require('disabled' in deep.lower(), 'voicebank deep training must remain disabled')
wave=(root/'src/yuaz_ddsp_resampler/neural_waveform.py')
trainer=(root/'src/yuaz_ddsp_resampler/train_neural_waveform.py')
train_cmd=(root/'scripts/train-neural-waveform.command')
require(wave.is_file(), 'neural waveform module missing')
require(trainer.is_file(), 'neural waveform trainer missing')
require(train_cmd.is_file(), 'neural waveform training command missing')
wave_text=wave.read_text()
trainer_text=trainer.read_text()
require('class YuazNeuralWaveformDecoder' in wave_text, 'neural waveform decoder class')
require('build_neural_conditioning' in wave_text, 'neural conditioning builder')
require('native-pitch reconstruction + alias-isolated multipitch cross-recording' in trainer_text, 'neural training definition')
require('PITCH_BUCKETS' in trainer_text, 'multipitch bucket split')
PY
python3 -m compileall -q "$ROOT/src"
while IFS= read -r -d '' f; do bash -n "$f"; done < <(find "$ROOT" -type f -name '*.command' -not -path '*/previous_versions/*' -print0)
echo "0.3.0 self-test OK"
