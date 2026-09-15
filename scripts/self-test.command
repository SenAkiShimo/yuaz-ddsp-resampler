#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${ROOT}/.venv/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$ROOT" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])

def require(condition, label):
    if not condition:
        raise AssertionError(f"self-test failed: {label}")

require((root / "VERSION").read_text().strip() == "0.3.0", "VERSION is not 0.3.0")
client = (root / "src/yuaz_ddsp_resampler/client.py").read_text()
server = (root / "src/yuaz_ddsp_resampler/server.py").read_text()
controls = (root / "src/yuaz_ddsp_resampler/controls.py").read_text()
require('ENGINE_VERSION = "0.3.0"' in client, "client engine version")
require('DEFAULT_PORT = 47889' in client, "client production port")
require('ENGINE_VERSION = "0.3.0"' in server, "server engine version")

match = re.search(r'_CONTROL_RE = re\.compile\(\s*r"\(([^)]*)\)', controls)
require(match is not None, "control parser regex")
flags = set(match.group(1).split("|"))
require(flags == {"YM", "YD", "YH", "YT", "YB", "YV", "YG", "YO", "YF", "YX", "YP", "YR"}, "public control set")
for removed in ("YQ", "YA", "YN"):
    require(removed not in controls, f"removed test control {removed} still present")
    require(removed not in server, f"removed server test route {removed} still present")
require('raw_bypass: float = 0.0' in controls, "YR raw bypass field")
require('def raw_bypass_enabled' in controls, "YR raw bypass property")

ai_controls = (root / "src/yuaz_ddsp_resampler/ai_vocal_controls.py").read_text()
require('mask = (strength > 1e-6).to(c.dtype) * voiced' in ai_controls, "AI control voiced mask")
require('control_gate_mode": "source-active-voiced"' in ai_controls, "AI control gate mode")
require('mask = strength * voiced' not in ai_controls, "obsolete strength mask still present")
require('t_progress = t_amount * t_amount * (3.0 - 2.0 * t_amount)' in ai_controls, "YT learned progress")
require('return t_ds, t_da, t_dg' in ai_controls, "YT learned residual continuity")

vocal = (root / "src/yuaz_ddsp_resampler/vocal_controls.py").read_text()
for label, expected in (
    ("YT carrier", 'tension_scale = carrier("tension", 0.88)'),
    ("YG carrier", 'gender_scale = carrier("gender_formant", 0.85)'),
    ("YO carrier", 'mouth_scale = carrier("mouth", 0.95)'),
    ("YF F0-relative register", 'harmonic_order = hz / f0_env'),
):
    require(expected in vocal, label)

install = (root / "scripts/install-openutau-macos.command").read_text()
require("0.3.0" in install, "installer version")
require(".yuaz-0.2.8ai14" in install, "ai.14 state namespace")
require("preserve_ai14" in install, "ai.14 preservation")

required_files = (
    "src/yuaz_ddsp_resampler/neural_waveform.py",
    "src/yuaz_ddsp_resampler/neural_waveform_v4.py",
    "src/yuaz_ddsp_resampler/neural_runtime.py",
    "src/yuaz_ddsp_resampler/train_neural_waveform.py",
    "src/yuaz_ddsp_resampler/train_neural_waveform_v3.py",
    "src/yuaz_ddsp_resampler/train_neural_waveform_v4.py",
    "src/yuaz_ddsp_resampler/export_neural_waveform_ab.py",
    "scripts/train-neural-waveform.command",
    "scripts/train-neural-waveform-v3.command",
    "scripts/train-neural-waveform-v4.command",
    "scripts/export-neural-waveform-ab.command",
)
for rel in required_files:
    require((root / rel).is_file(), f"missing {rel}")

wave = (root / "src/yuaz_ddsp_resampler/neural_waveform.py").read_text()
wave_v4 = (root / "src/yuaz_ddsp_resampler/neural_waveform_v4.py").read_text()
runtime = (root / "src/yuaz_ddsp_resampler/neural_runtime.py").read_text()
v3 = (root / "src/yuaz_ddsp_resampler/train_neural_waveform_v3.py").read_text()
v4 = (root / "src/yuaz_ddsp_resampler/train_neural_waveform_v4.py").read_text()
require("class YuazNeuralWaveformDecoder" in wave, "neural waveform decoder")
require("build_neural_conditioning" in wave, "neural conditioning")
require("SOURCE_DETAIL_CHANNELS = SOURCE_DETAIL_BANDS * 2 + 3" in wave_v4, "v4 source detail width")
require("build_pitch_invariant_source_detail" in wave_v4, "v4 source detail extractor")
require("class NeuralWaveformRuntimeRoute" in runtime, "neural runtime route")
require("voicebank-mismatch" in runtime, "voicebank provenance guard")
require("trainer_generation\": \"conditioned-v3\"" in v3, "v3 metadata")
require("trainer_generation\": \"conditioned-v4\"" in v4, "v4 metadata")
require("_V3_BASE_LOSS = v3.neural_waveform_loss_v3" in v4, "v4 frozen base loss")
require("State.neural_route.select_record" in server, "server neural route selection")
require("response.update(State.neural_route.stats())" in server, "server neural diagnostics")
PY
python3 -m compileall -q "$ROOT/src"
while IFS= read -r -d '' f; do bash -n "$f"; done < <(find "$ROOT" -type f -name '*.command' -not -path '*/previous_versions/*' -print0)
echo "0.3.0 self-test OK"
