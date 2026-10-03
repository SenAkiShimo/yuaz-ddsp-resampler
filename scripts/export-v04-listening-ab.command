#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

[ -x .venv/bin/python ] || { echo "Run setup-macos.command first."; exit 1; }
[ -f config.json ] || { echo "Run configure-macos.command first."; exit 1; }

VOICEBANK="${1:-}"
if [ -z "$VOICEBANK" ]; then
  echo "Drop the voicebank folder here, then press Return:"
  read -r VOICEBANK
else
  shift
fi

VOICEBANK="${VOICEBANK#\'}"; VOICEBANK="${VOICEBANK%\'}"
VOICEBANK="${VOICEBANK#\"}"; VOICEBANK="${VOICEBANK%\"}"
[ -d "$VOICEBANK" ] || { echo "Voicebank folder not found: $VOICEBANK"; exit 1; }

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Exporting matched blind V3/V4/HiFi listening set."
echo "Default pitch shifts: 0, +5, +9, +14 semitones."
echo "The command will refuse checkpoints trained on a different voicebank."

exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.export_v04_listening_ab \
  --project-root "$ROOT" \
  --voicebank "$VOICEBANK" \
  "$@"
