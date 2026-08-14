#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "===== 0.2.8ai.12 DOCTOR ====="
echo "Version: $(cat VERSION 2>/dev/null || echo missing)"
[ -f config.json ] || { echo "FAIL: config.json missing"; exit 1; }
python3 - config.json <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
print('engine_version:', c.get('engine_version'))
print('runtime ID:', c.get('runtime_id'))
print('host/port:', c.get('host'), c.get('port'))
print('state namespace:', c.get('state_namespace'))
print('migration predecessor 0.2.8ai.11:', c.get('previous_028ai11_readonly_state_namespace'))
print('migration predecessor 0.2.8ai.10:', c.get('previous_028ai10_readonly_state_namespace'))
print('analysis/synthesis sample rates:', 24000, c.get('ddsp_synthesis_sr'))
print('full-band crossover:', c.get('ddsp_fullband_crossover_start_hz'), c.get('ddsp_fullband_crossover_full_hz'))
print('upper-band parameter head:', c.get('ai12_upperband_head_enabled'), c.get('ai12_upperband_head_start_hz'), c.get('ai12_upperband_head_full_hz'))
assert c.get('engine_version')=='0.2.8ai.12'
assert int(c.get('port'))==47885
assert c.get('runtime_id')=='yuaz-0.2.8ai.12-control-v12'
assert c.get('state_namespace')=='.yuaz-0.2.8ai12'
assert c.get('previous_028ai11_readonly_state_namespace')=='.yuaz-0.2.8ai11'
assert c.get('previous_028ai10_readonly_state_namespace')=='.yuaz-0.2.8ai10'
assert c.get('previous_028ai9_readonly_state_namespace')=='.yuaz-0.2.8ai9'
assert int(c.get('ddsp_synthesis_sr'))==48000
assert float(c.get('ddsp_fullband_crossover_start_hz'))==8800.0
assert float(c.get('ddsp_fullband_crossover_full_hz'))==12100.0
assert c.get('ai12_upperband_head_enabled') is True
assert float(c.get('ai12_upperband_head_start_hz'))==8400.0
assert float(c.get('ai12_upperband_head_full_hz'))==12400.0
assert c.get('previous_028ai7_readonly_state_namespace')=='.yuaz-0.2.8ai7'
PY
[ -x .venv/bin/python ] || { echo "FAIL: environment missing"; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python - <<'PY'
import torch,numpy,librosa,soundfile,yaml
from yuaz_ddsp_resampler.ai_vocal_controls import AIControlAdapter
from yuaz_ddsp_resampler.highband_foundation import HighBandFoundation
print('Pinned packages:', torch.__version__, numpy.__version__, librosa.__version__, soundfile.__version__, yaml.__version__)
print('Control adapter parameters:', sum(p.numel() for p in AIControlAdapter().parameters()))
print('High-Band Foundation legacy-v1 parameters:', sum(p.numel() for p in HighBandFoundation().parameters())); print('High-Band Foundation v2 parameters:', sum(p.numel() for p in HighBandFoundation(hidden=40,dilations=(1,2,4,8,16,32,64,128)).parameters()))
PY
echo "Port 47885 listener:"
lsof -nP -iTCP:47885 -sTCP:LISTEN 2>/dev/null || echo "  none (normal before first render)"
WRAP="$HOME/Library/OpenUtau/Resamplers/Yuaz-DDSP-Resampler-v0.2.8ai.12.sh"
[ -f "$WRAP" ] && echo "PASS: current wrapper installed" || echo "Current wrapper not installed yet."
if [ -d "$HOME/Library/OpenUtau/Resamplers" ]; then
  OLD="$(find "$HOME/Library/OpenUtau/Resamplers" -maxdepth 1 -type f \( -name 'Yuaz-DDSP-Resampler*.sh' -o -name 'Yuaz-DDSP-Resampler*.yaml' \) ! -name 'Yuaz-DDSP-Resampler-v0.2.8ai.12.sh' ! -name 'Yuaz-DDSP-Resampler-v0.2.8ai.12.yaml' -print)"
  if [ -n "$OLD" ]; then
    echo "WARNING: old Yuaz wrappers remain:"
    echo "$OLD"
  else
    echo "PASS: no previous Yuaz wrappers remain."
  fi
fi
if [ -f "$ROOT/control_models/highband_foundation-v2.pt" ]; then
  echo "PASS: High-Band Foundation v2 present: $ROOT/control_models/highband_foundation-v2.pt"
elif [ -f "$ROOT/control_models/highband_foundation-v1.pt" ]; then
  echo "PASS: legacy High-Band Foundation v1 present; runtime hybrid continuity fix is active."
else
  echo "No package-level Foundation checkpoint found. This is normal when r1/r2 is pinned inside migrated voicebank state; YH can still load it from the voicebank registry."
fi
echo "PASS: ai.12 upper-band parameter head enabled on the 48 kHz DDSP body; ai.11 synthesis/mixer remains available as fallback. YH stays a refinement layer."
echo "Doctor OK"
