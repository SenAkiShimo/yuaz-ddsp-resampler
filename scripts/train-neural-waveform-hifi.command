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
MANIFEST="$("$ROOT/.venv/bin/python" - "$VOICEBANK" <<'PY'
import sys
from pathlib import Path
from yuaz_ddsp_resampler.state import resolve_ai_state
bank = Path(sys.argv[1]).expanduser().resolve()
state, info = resolve_ai_state(bank, verify=True)
if state is None:
    raise SystemExit("No valid .yuaz-0.2.8ai14 generation found for this voicebank.")
manifest = Path(state) / "manifest.json"
if not manifest.is_file():
    raise SystemExit(f"Active ai.14 generation has no manifest: {manifest}")
print(manifest)
PY
)"

echo "v0.4 HiFi high-band fidelity fine-tune"
echo "Voicebank: $VOICEBANK"
echo "Manifest:  $MANIFEST"
echo "Warm-start: this voicebank's robust v4 Pareto checkpoint"
echo "High-band conditioning: 6-22 kHz, 64 fine bands, no frequency smoothing"
echo "Default schedule: 1 native epoch + 2 multipitch epochs"

exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.train_neural_waveform_hifi \
  --project-root "$ROOT" \
  --voicebank "$VOICEBANK" \
  --manifest "$MANIFEST" \
  "$@"
