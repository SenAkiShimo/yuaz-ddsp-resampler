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

LIMIT="${YUAZ_CLARITY_LIMIT:-96}"
EPOCHS="${YUAZ_CLARITY_EPOCHS:-4}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.train_clarity_refiner \
  "$VOICEBANK" \
  --project-root "$ROOT" \
  --limit "$LIMIT" \
  --epochs "$EPOCHS"

open "$ROOT/clarity-refiner-output/examples" 2>/dev/null || true
