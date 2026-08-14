#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${YUAZ_CONTROL_DATASETS:-$HOME/YuazControlDatasets}"
PHONATION="${YUAZ_PHONATION_ROOT:-$DATA_ROOT/PhonationModesOSF}"
MOCHA="${YUAZ_MOCHA_ROOT:-$DATA_ROOT/MOCHA-TIMIT}"
CACHE="$DATA_ROOT/_yuaz_ai_cache/phonation-osf-mocha-v1"
OUT="$ROOT/control_models/ai_phonation_foundation-v1.pt"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "Run setup-macos.command first."; exit 1; }
[ -f "$PHONATION/.yuaz-phonation-modes-osf-manifest.json" ] || { echo "OSF Phonation Modes not ready: $PHONATION"; exit 1; }
[ -f "$MOCHA/.yuaz-mocha-ready.json" ] || { echo "MOCHA not validated/extracted: $MOCHA"; exit 1; }
mkdir -p "$CACHE" "$ROOT/control_models"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
SHA="$($PY - <<PY
import json
from pathlib import Path
from yuaz_ddsp_resampler.state import sha256
c=json.loads(Path('$ROOT/config.json').read_text()); print(sha256(Path(c['checkpoint']).expanduser()))
PY
)"
echo "Yuaz AI Phonation Foundation v1 — YT Tension + YV Voicing/Closure"
echo "OSF Phonation Modes: direct singing breathy/modal/pressed supervision"
echo "MOCHA: laryngograph periodicity auxiliary supervision"
"$PY" -m yuaz_ddsp_resampler.multimodal_control_training prepare-phonation "$PHONATION" "$CACHE" --project-root "$ROOT"
"$PY" -m yuaz_ddsp_resampler.multimodal_control_training prepare-mocha "$MOCHA" "$CACHE" --project-root "$ROOT"
"$PY" -m yuaz_ddsp_resampler.multimodal_control_training train-phonation "$CACHE/phonation_direct_shards" "$CACHE/phonation_shards" "$OUT" --checkpoint-sha "$SHA" --epochs 12
B="$HOME/Documents/Yuaz-DDSP-Backups/control-models";mkdir -p "$B";cp "$OUT" "$B/ai_phonation_foundation-v1-PhonationModes-MOCHA.pt";[ -f "$OUT.json" ] && cp "$OUT.json" "$B/ai_phonation_foundation-v1-PhonationModes-MOCHA.pt.json" || true
echo "Phonation foundation ready: $OUT"
