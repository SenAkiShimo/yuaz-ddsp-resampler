#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -x .venv/bin/python ]; then
  echo "Run scripts/setup-macos.command first."
  exit 1
fi

strip_path() {
  python3 - "$1" <<'PY'
import shlex, sys
parts = shlex.split(sys.argv[1].strip())
print(parts[0] if parts else "")
PY
}

echo "Drop the voicebank folder here, then press Return:"
read -r RAW
VOICEBANK="$(strip_path "$RAW")"
if [ ! -d "$VOICEBANK" ]; then
  echo "Voicebank folder not found: $VOICEBANK"
  exit 1
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.source_high_detail "$VOICEBANK"
