#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -x .venv/bin/python ] || { echo "Run setup-macos.command first."; exit 1; }
strip_path() { python3 - "$1" <<'PY'
import shlex,sys
p=shlex.split(sys.argv[1].strip()); print(p[0] if p else '')
PY
}
YUAZ="${YUAZ_SGR_ROOT:-$HOME/Downloads/yuaz-sgr}"
if [ ! -d "$YUAZ/yuaz_sgr/models" ]; then
  echo "Drop the Yuaz SGR repository folder here, then press Return:"
  read -r RAW; YUAZ="$(strip_path "$RAW")"
fi
[ -d "$YUAZ/yuaz_sgr/models" ] || { echo "Yuaz SGR model directory not found: $YUAZ/yuaz_sgr/models"; exit 1; }
CKPT="${YUAZ_SGR_CHECKPOINT:-}"
if [ -z "$CKPT" ]; then
  for C in "$HOME/Downloads/model_351000_resampler.pt" "$HOME/Downloads/model_351000.pt" "$YUAZ/yuaz_sgr_inference/checkpoint_300k.pt"; do
    if [ -f "$C" ]; then CKPT="$C"; break; fi
  done
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "Drop a compatible Yuaz checkpoint (.pt) here, then press Return:"
  read -r RAW; CKPT="$(strip_path "$RAW")"
fi
[ -f "$CKPT" ] || { echo "Checkpoint not found: $CKPT"; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
echo "Probing/importing Yuaz checkpoint..."
IMPORT_OUT="$ROOT/.checkpoint-import.json"
"$ROOT/.venv/bin/python" - "$YUAZ" "$CKPT" "$IMPORT_OUT" <<'PY'
import json,sys
from pathlib import Path
from yuaz_ddsp_resampler.checkpoint_registry import import_checkpoint
m=import_checkpoint(sys.argv[2],sys.argv[1])
Path(sys.argv[3]).write_text(json.dumps(m,indent=2),encoding='utf-8')
print('Imported:',m['model_id'],'step=',m.get('source_step'),'runtime=',m['runtime_path'])
PY
APP_STATE="$HOME/Library/Application Support/YuazDDSP/state"
mkdir -p "$APP_STATE"
"$ROOT/.venv/bin/python" - "$YUAZ" "$IMPORT_OUT" "$APP_STATE/voicebank_registry-0.2.8ai14.json" <<'PY'
import json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[2]).read_text())
config={
 "yuaz_repo":str(Path(sys.argv[1]).resolve()), "checkpoint":m['runtime_path'],
 "base_checkpoint_model_id":m['model_id'], "base_checkpoint_source_name":m.get('source_checkpoint'),
 "base_checkpoint_sha256":m['source_checkpoint_sha256'], "base_checkpoint_runtime_sha256":m['runtime_sha256'],
 "base_checkpoint_step":m.get('source_step'), "base_checkpoint_registry":str(Path(m['runtime_path']).parent.parent/'registry.json'),
 "host":"127.0.0.1", "port":47889, "engine_version":"0.3.0", "runtime_id":"yuaz-0.3.0",
 "acoustic_base":"0.2.8ai.14-state-plus-0.3.0-neural-waveform-runtime", "transition_ms":70.0, "use_rvq":False,
 "output_sr":44100, "ddsp_synthesis_sr":48000, "ddsp_fullband_crossover_start_hz":8800.0,
 "ddsp_fullband_crossover_full_hz":12100.0, "ai12_upperband_head_enabled":True,
 "ai12_upperband_head_start_hz":8400.0, "ai12_upperband_head_full_hz":12400.0,
 "ai13_upperband_guard_enabled":True, "ai13_upperband_head_start_hz":8200.0, "ai13_upperband_head_full_hz":13800.0,
 "normalize_voicebank_loudness":True, "normalization_target_dbfs":-18.0, "normalization_peak_ceiling_dbfs":-1.0,
 "normalization_peak_guard_knee_db":3.0, "normalization_emergency_max_abs_gain_db":30.0, "normalization_tolerance_db":0.05,
 "enable_fidelity_refiner":True, "fidelity_residual_hard_limit":0.085,
 "registry_path":str(Path(sys.argv[3]).expanduser().resolve()),
 "state_namespace":".yuaz-0.2.8ai14", "state_access":"read-only-ai14-compatibility",
 "preserve_ai14":True, "allow_029_voicebank_training":False,
 "trained_artifact_suffix":".ai14", "ai_control_model_policy":"checkpoint-matched-only"
}
Path('config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
print('Wrote config.json')
print('Base model:',config['base_checkpoint_model_id'],'step',config['base_checkpoint_step'])
print('ai.14 state: READ-ONLY')
print('0.3.0 port:',config['port'])
PY
rm -f "$IMPORT_OUT"
chmod +x yuaz-ddsp-resampler scripts/*.command *.command 2>/dev/null || true
