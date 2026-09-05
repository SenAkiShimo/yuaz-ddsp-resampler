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

CANDIDATES=(
  "$ROOT/control_models/neural-waveform-v0.3.0-conditioned-v3-pareto-best.pt"
  "$ROOT/control_models/neural-waveform-v0.3.0-conditioned-v3-multipitch-best.pt"
  "$ROOT/control_models/neural-waveform-v0.3.0-conditioned-v3.pt"
  "$ROOT/control_models/neural-waveform-v0.3.0-conditioned-multipitch-best.pt"
  "$ROOT/control_models/neural-waveform-v0.3.0-conditioned.pt"
  "$ROOT/control_models/neural-waveform-v0.3.0-multipitch-best.pt"
  "$ROOT/control_models/neural-waveform-v0.3.0.pt"
)
CHECKPOINT=""
for CANDIDATE in "${CANDIDATES[@]}"; do
  if [ -f "$CANDIDATE" ]; then
    CHECKPOINT="$CANDIDATE"
    break
  fi
done
[ -n "$CHECKPOINT" ] || { echo "No neural waveform checkpoint found."; exit 1; }

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
echo "A/B checkpoint: $CHECKPOINT"
exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.export_neural_waveform_ab \
  --project-root "$ROOT" \
  --voicebank "$VOICEBANK" \
  --checkpoint "$CHECKPOINT" \
  "${@:2}"
