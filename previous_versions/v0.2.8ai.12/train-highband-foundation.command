#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${YUAZ_CONTROL_DATASETS:-$HOME/YuazControlDatasets}"
WORK="$DATA_ROOT/HighBandFoundation"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "Run ./setup-macos.command first."; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
[ -f "$WORK/shards/manifest.json" ] || { echo "Missing paired shards. Run ./prepare-highband-training.command first."; exit 1; }
mkdir -p "$ROOT/control_models" "$HOME/Documents/Yuaz-DDSP-Backups/control-models/0.2.8ai.12"
EPOCHS="${YUAZ_HIGHBAND_EPOCHS:-10}"
BATCH="${YUAZ_HIGHBAND_BATCH:-4}"
OUT="$ROOT/control_models/highband_foundation-v2.pt"
echo "Yuaz 0.2.8ai.12 high-band hotfix — Train High-Band Foundation v2"
echo "Architecture: wider-context waveform residual BWE network"
echo "Input: 48 kHz audio degraded through a 24 kHz bottleneck"
echo "Target: the same original full-band singing"
echo "Loss: phase-tolerant multi-resolution magnitude + temporal band envelope + light waveform residual."
echo "Low-F0 validation is weighted explicitly."
echo
"$PY" -m yuaz_ddsp_resampler.highband_training train \
  --manifest "$WORK/shards/manifest.json" --out "$OUT" \
  --epochs "$EPOCHS" --batch-size "$BATCH" --device auto
cp "$OUT" "$HOME/Documents/Yuaz-DDSP-Backups/control-models/0.2.8ai.12/highband_foundation-v2.pt"
shasum -a 256 "$OUT"
echo
echo "High-Band Foundation ready: $OUT"
echo "Next: ./probe-highband-foundation.command"
