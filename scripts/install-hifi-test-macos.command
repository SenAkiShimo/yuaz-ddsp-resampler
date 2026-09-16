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

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
BANK_ID="$(.venv/bin/python - "$VOICEBANK" <<'PY'
import sys
from yuaz_ddsp_resampler.train_neural_waveform_v04 import voicebank_id
print(voicebank_id(sys.argv[1]))
PY
)"
REL="control_models/v0.4/$BANK_ID/hifi/neural-waveform-v0.4-hifi-pareto-best.pt"
CHECKPOINT="$ROOT/$REL"
[ -f "$CHECKPOINT" ] || { echo "HiFi Pareto checkpoint not found: $CHECKPOINT"; exit 1; }

BACKUP="$(mktemp /tmp/yuaz-config-hifi.XXXXXX.json)"
cp config.json "$BACKUP"
restore_config() {
  cp "$BACKUP" config.json
  rm -f "$BACKUP"
}
trap restore_config EXIT

.venv/bin/python - "$ROOT/config.json" "$REL" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
rel = sys.argv[2]
data = json.loads(path.read_text(encoding="utf-8"))
data["neural_waveform_enabled"] = True
data["neural_waveform_checkpoint"] = rel
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Temporary install checkpoint: {rel}")
PY

bash "$ROOT/scripts/install-openutau-macos.command"

echo ""
echo "HiFi test runtime installed for: $VOICEBANK"
echo "Checkpoint: $CHECKPOINT"
echo "The source config.json has been restored; only the installed runtime is pinned to this HiFi checkpoint."
echo "Restart OpenUtau before rendering the A/B test."
