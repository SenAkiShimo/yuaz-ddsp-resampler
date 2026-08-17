#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${YUAZ_CONTROL_DATASETS:-$HOME/YuazControlDatasets}"
PHONATION="${YUAZ_PHONATION_ROOT:-$DATA_ROOT/PhonationModesOSF}"
MOCHA="${YUAZ_MOCHA_ROOT:-$DATA_ROOT/MOCHA-TIMIT}"
OUT="$ROOT/control_models/ai_phonation_foundation-v1.pt"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "Run setup-macos.command first."; exit 1; }
[ -f "$PHONATION/.yuaz-phonation-modes-osf-manifest.json" ] || { echo "OSF Phonation Modes not ready: $PHONATION"; exit 1; }
[ -f "$MOCHA/.yuaz-mocha-ready.json" ] || { echo "MOCHA not validated/extracted: $MOCHA"; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
SHA="$($PY - <<PY
import json
from pathlib import Path
from yuaz_ddsp_resampler.checkpoint_identity import checkpoint_identity_sha
c=json.loads(Path('$ROOT/config.json').read_text()); print(checkpoint_identity_sha(Path(c['checkpoint']).expanduser()))
PY
)"
CACHE="$DATA_ROOT/_yuaz_ai_cache/phonation-osf-mocha-v1-${SHA:0:16}"
rm -rf "$CACHE"
mkdir -p "$CACHE" "$ROOT/control_models"
"$PY" -m yuaz_ddsp_resampler.multimodal_control_training prepare-phonation "$PHONATION" "$CACHE" --project-root "$ROOT"
"$PY" -m yuaz_ddsp_resampler.multimodal_control_training prepare-mocha "$MOCHA" "$CACHE" --project-root "$ROOT"
"$PY" -m yuaz_ddsp_resampler.multimodal_control_training train-phonation "$CACHE/phonation_direct_shards" "$CACHE/phonation_shards" "$OUT" --checkpoint-sha "$SHA" --epochs 12
B="$HOME/Documents/Yuaz-DDSP-Backups/control-models"
mkdir -p "$B"
cp "$OUT" "$B/ai_phonation_foundation-v1-PhonationModes-MOCHA-${SHA:0:16}.pt"
[ -f "$OUT.json" ] && cp "$OUT.json" "$B/ai_phonation_foundation-v1-PhonationModes-MOCHA-${SHA:0:16}.pt.json" || true
printf '%s\n' "$OUT"