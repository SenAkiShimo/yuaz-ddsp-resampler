#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -x .venv/bin/python ]; then
  echo "Run scripts/setup-macos.command first."
  exit 1
fi
if [ ! -f config.json ]; then
  echo "Run scripts/configure-macos.command first."
  exit 1
fi

strip_path() {
  python3 - "$1" <<'PY'
import shlex, sys
parts = shlex.split(sys.argv[1].strip())
print(parts[0] if parts else "")
PY
}

echo "Drop the prepared voicebank folder here, then press Return:"
read -r RAW
VOICEBANK="$(strip_path "$RAW")"
if [ ! -d "$VOICEBANK" ]; then
  echo "Voicebank folder not found: $VOICEBANK"
  exit 1
fi

PAIRS="${YUAZ_HIGH_DETAIL_PAIRS:-96}"
EPOCHS="${YUAZ_HIGH_DETAIL_EPOCHS:-6}"
LR="${YUAZ_HIGH_DETAIL_LR:-0.0004}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.train_high_detail_tf \
  "$VOICEBANK" \
  --project-root "$ROOT" \
  --pairs "$PAIRS" \
  --epochs "$EPOCHS" \
  --lr "$LR"

MODEL="$ROOT/high-detail-tf-output/high_detail_tf.pt"
if [ -f "$MODEL" ]; then
  STATE="$VOICEBANK/.yuaz-0.2.8ai14"
  mkdir -p "$STATE"
  cp "$MODEL" "$STATE/high_detail_tf.pt"
  echo "Installed TF model: $STATE/high_detail_tf.pt"
fi

open "$ROOT/high-detail-tf-output/examples" 2>/dev/null || true
