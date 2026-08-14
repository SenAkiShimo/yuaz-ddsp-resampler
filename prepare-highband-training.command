#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${YUAZ_CONTROL_DATASETS:-$HOME/YuazControlDatasets}"
WORK="$DATA_ROOT/HighBandFoundation"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "Run ./setup-macos.command first."; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
[ -f "$WORK/audit.json" ] || { echo "Missing $WORK/audit.json. Run ./audit-highband-datasets.command first."; exit 1; }
SEGMENTS="${YUAZ_HIGHBAND_SEGMENTS:-6000}"
VAL="${YUAZ_HIGHBAND_VAL_SEGMENTS:-800}"
echo "Yuaz 0.2.8ai.13 — Prepare paired High-Band shards"
echo "Train segments: $SEGMENTS"
echo "Validation segments: $VAL"
echo "Low-F0 records are oversampled automatically."
echo
"$PY" -m yuaz_ddsp_resampler.highband_training prepare \
  --audit "$WORK/audit.json" --out-dir "$WORK/shards" \
  --segments "$SEGMENTS" --val-segments "$VAL"
echo
echo "Next: ./train-highband-foundation.command"
