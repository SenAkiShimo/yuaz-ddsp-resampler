#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -x .venv/bin/python ]; then
  echo "Run scripts/setup-macos.command first."
  exit 1
fi
echo "LEGACY DIAGNOSTIC ONLY: this creates statistical technique profiles and is NOT the realtime AI controller."
echo "For 0.2.8ai.11 learned controls, use train-ai-control-foundation.command instead."
echo
echo "Drop the extracted GTSinger root folder here, then press Return:"
read -r RAW
DATASET="${RAW%\'}"; DATASET="${DATASET#\'}"; DATASET="${DATASET%\"}"; DATASET="${DATASET#\"}"
if [ ! -d "$DATASET" ]; then
  echo "Dataset folder not found: $DATASET"
  exit 1
fi
OUT="$ROOT/control_training/technique_profiles.npz"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m yuaz_ddsp_resampler.control_training gtsinger "$DATASET" "$OUT"
echo
echo "Created:"
echo "  $OUT"
echo "  ${OUT%.npz}.json"
