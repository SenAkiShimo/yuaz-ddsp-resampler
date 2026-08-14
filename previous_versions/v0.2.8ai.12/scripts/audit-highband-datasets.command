#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${YUAZ_CONTROL_DATASETS:-$HOME/YuazControlDatasets}"
WORK="$DATA_ROOT/HighBandFoundation"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "Run ./setup-macos.command first."; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
GTS="$DATA_ROOT/GTSinger"
if [ -f "$DATA_ROOT/ACTIVE_GTSINGER_ROOT.txt" ]; then GTS="$(cat "$DATA_ROOT/ACTIVE_GTSINGER_ROOT.txt")"; fi
VOC="$DATA_ROOT/VocalSetMirror"
PHO="$DATA_ROOT/PhonationModesOSF"
TOOLS="$HOME/Library/Application Support/YuazDDSP/training-tools-gender/site-packages"
mkdir -p "$WORK/vocalset_cache"

if find "$VOC" -type f -name '*.parquet' -print -quit 2>/dev/null | grep -q .; then
  if ! PYTHONPATH="$TOOLS${PYTHONPATH:+:$PYTHONPATH}" "$PY" - <<'PYARROW' >/dev/null 2>&1
import pyarrow
PYARROW
  then
    echo "Installing developer-only pyarrow from the Tsinghua PyPI mirror..."
    mkdir -p "$TOOLS"
    "$PY" -m pip install --target "$TOOLS" pyarrow -i https://pypi.tuna.tsinghua.edu.cn/simple
  fi
fi

echo "Yuaz 0.2.8ai.12 — High-Band Foundation bandwidth audit"
echo "This does not download new datasets. It scans the data already on this Mac."
echo
echo "GTSinger:       $GTS"
echo "VocalSet:       $VOC"
echo "PhonationModes: $PHO"
echo "Output:         $WORK/audit.json"
echo
PYTHONPATH="$TOOLS${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m yuaz_ddsp_resampler.highband_training audit \
  --gtsinger "$GTS" --vocalset "$VOC" --phonation "$PHO" \
  --vocalset-cache "$WORK/vocalset_cache" --out "$WORK/audit.json"
echo
echo "Audit ready: $WORK/audit.json"
echo "Next: ./prepare-highband-training.command"
