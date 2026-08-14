#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${YUAZ_CONTROL_DATASETS:-$HOME/YuazControlDatasets}";MOCHA="${YUAZ_MOCHA_ROOT:-$DATA_ROOT/MOCHA-TIMIT}";CACHE="$DATA_ROOT/_yuaz_ai_cache/phonation-osf-mocha-v1";OUT="$ROOT/control_models/ai_mouth_foundation-v1.pt";PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "Run setup-macos.command first."; exit 1; };[ -d "$MOCHA/extracted" ] || { echo "MOCHA not ready: $MOCHA"; exit 1; };mkdir -p "$CACHE" "$ROOT/control_models";export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
SHA="$($PY - <<PY
import json
from pathlib import Path
from yuaz_ddsp_resampler.state import sha256
c=json.loads(Path('$ROOT/config.json').read_text()); print(sha256(Path(c['checkpoint']).expanduser()))
PY
)"
echo "Yuaz AI Mouth Foundation v1 — YO Mouth/Resonance"
[ -d "$CACHE/mouth_shards" ] || "$PY" -m yuaz_ddsp_resampler.multimodal_control_training prepare-mocha "$MOCHA" "$CACHE" --project-root "$ROOT"
"$PY" -m yuaz_ddsp_resampler.multimodal_control_training train-mouth "$CACHE/mouth_shards" "$OUT" --checkpoint-sha "$SHA" --epochs 12
B="$HOME/Documents/Yuaz-DDSP-Backups/control-models";mkdir -p "$B";cp "$OUT" "$B/ai_mouth_foundation-v1-MOCHA.pt";[ -f "$OUT.json" ] && cp "$OUT.json" "$B/ai_mouth_foundation-v1-MOCHA.pt.json" || true
echo "Mouth foundation ready: $OUT"
