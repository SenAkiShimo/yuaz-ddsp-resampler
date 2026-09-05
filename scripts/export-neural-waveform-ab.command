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
fi
VOICEBANK="${VOICEBANK#\'}"; VOICEBANK="${VOICEBANK%\'}"
VOICEBANK="${VOICEBANK#\"}"; VOICEBANK="${VOICEBANK%\"}"
[ -d "$VOICEBANK" ] || { echo "Voicebank folder not found: $VOICEBANK"; exit 1; }

BEST_CONDITIONED="$ROOT/control_models/neural-waveform-v0.3.0-conditioned-multipitch-best.pt"
FINAL_CONDITIONED="$ROOT/control_models/neural-waveform-v0.3.0-conditioned.pt"
BEST_LEGACY="$ROOT/control_models/neural-waveform-v0.3.0-multipitch-best.pt"
FINAL_LEGACY="$ROOT/control_models/neural-waveform-v0.3.0.pt"
if [ -f "$BEST_CONDITIONED" ]; then
  CHECKPOINT="$BEST_CONDITIONED"
elif [ -f "$FINAL_CONDITIONED" ]; then
  CHECKPOINT="$FINAL_CONDITIONED"
elif [ -f "$BEST_LEGACY" ]; then
  CHECKPOINT="$BEST_LEGACY"
else
  CHECKPOINT="$FINAL_LEGACY"
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
echo "A/B checkpoint: $CHECKPOINT"
exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.export_neural_waveform_ab \
  --project-root "$ROOT" \
  --voicebank "$VOICEBANK" \
  --checkpoint "$CHECKPOINT" \
  "${@:2}"
