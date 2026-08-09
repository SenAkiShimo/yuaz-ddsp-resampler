#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
strip_path() {
  python3 - "$1" <<'PY'
import shlex, sys
parts = shlex.split(sys.argv[1].strip())
print(parts[0] if parts else "")
PY
}
YUAZ="${YUAZ_SGR_ROOT:-$HOME/Downloads/yuaz-sgr}"
if [ ! -d "$YUAZ/yuaz_sgr/models" ]; then
  echo "Drop the Yuaz SGR repository folder here, then press Return:"
  read -r RAW
  YUAZ="$(strip_path "$RAW")"
fi
if [ ! -d "$YUAZ/yuaz_sgr/models" ]; then
  echo "Yuaz SGR model directory not found: $YUAZ/yuaz_sgr/models"
  exit 1
fi
CKPT="${YUAZ_SGR_CHECKPOINT:-$YUAZ/yuaz_sgr_inference/checkpoint_300k.pt}"
if [ ! -f "$CKPT" ]; then
  echo "Drop checkpoint_300k.pt here, then press Return:"
  read -r RAW
  CKPT="$(strip_path "$RAW")"
fi
if [ ! -f "$CKPT" ]; then
  echo "Checkpoint not found: $CKPT"
  exit 1
fi
python3 - "$YUAZ" "$CKPT" <<'PY'
import json, sys
from pathlib import Path
config = {
    "yuaz_repo": str(Path(sys.argv[1]).resolve()),
    "checkpoint": str(Path(sys.argv[2]).resolve()),
    "host": "127.0.0.1",
    "port": 47860,
    "engine_version": "0.2.7-alpha.1",
    "transition_ms": 70.0,
    "use_rvq": False,
    "output_sr": 44100,
    "normalize_voicebank_loudness": True,
    "normalization_target_dbfs": -18.0,
    "normalization_peak_ceiling_dbfs": -1.0,
    "normalization_peak_guard_knee_db": 3.0,
    "normalization_emergency_max_abs_gain_db": 30.0,
    "normalization_tolerance_db": 0.05,
    "registry_path": str((Path.cwd() / "voicebank_registry.json").resolve()),
}
Path("config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
print("Wrote config.json")
PY
chmod +x yuaz-ddsp-resampler scripts/*.command
