#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${YUAZ_CONTROL_DATASETS:-$HOME/YuazControlDatasets}"
TOOLS="$HOME/Library/Application Support/YuazDDSP/training-tools-gender/site-packages"
MARKER="$DATA_ROOT/ACTIVE_VOCALSET_ROOT.txt"
[ -x "$ROOT/.venv/bin/python" ] || { echo "Run setup-macos.command first."; exit 1; }
[ -f "$MARKER" ] || { echo "Run setup-gender-training.command first."; exit 1; }
VOCALSET="$(cat "$MARKER")"
[ -d "$VOCALSET/data" ] || { echo "VocalSet parquet data not found: $VOCALSET/data"; exit 1; }
CACHE="$DATA_ROOT/_yuaz_ai_cache/vocalset-gender-ddsp-v1"
OUT="$ROOT/control_models/ai_gender_foundation-v1.pt"
mkdir -p "$CACHE" "$ROOT/control_models"
export PYTHONPATH="$TOOLS:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Yuaz AI Gender/Formant Foundation v1"
echo "Source: VocalSet straight singing only"
echo "Target: speaker-aggregate female<->male Yuaz spectral-envelope direction"
echo "Guard: spectral-only; AP/gate outputs disabled; speaker-disjoint validation"
echo
"$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.gender_control_training prepare "$VOCALSET" "$CACHE" --project-root "$ROOT"
"$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.gender_control_training train "$CACHE/shards" "$OUT" --epochs 12
BACKUP="$HOME/Documents/Yuaz-DDSP-Backups/control-models"
mkdir -p "$BACKUP"
cp "$OUT" "$BACKUP/ai_gender_foundation-v1-VocalSet.pt"
[ -f "$OUT.json" ] && cp "$OUT.json" "$BACKUP/ai_gender_foundation-v1-VocalSet.pt.json" || true
echo
echo "Gender foundation ready: $OUT"
echo "Backup copy: $BACKUP/ai_gender_foundation-v1-VocalSet.pt"
echo "Next: run deep-train-voicebank.command only when you want 0.2.8ai.13 to pin this pack into its own isolated generation."
