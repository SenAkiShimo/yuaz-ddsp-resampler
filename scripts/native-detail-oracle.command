#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -x .venv/bin/python ]; then
  echo "Run scripts/setup-macos.command first."
  exit 1
fi

DUMP="${1:-$HOME/Desktop/YuazClarityDump}"
STRENGTH="${YUAZ_ORACLE_STRENGTH:-1.0}"

if [ ! -f "$DUMP/00_source_native.wav" ]; then
  echo "Missing: $DUMP/00_source_native.wav"
  exit 1
fi
if [ ! -f "$DUMP/07_final.wav" ]; then
  echo "Missing: $DUMP/07_final.wav"
  exit 1
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.native_detail_oracle \
  "$DUMP" \
  --strength "$STRENGTH"

open "$DUMP/oracle" 2>/dev/null || true
