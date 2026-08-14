#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${YUAZ_CONTROL_DATASETS:-$HOME/YuazControlDatasets}"
DEST="$DATA_ROOT/GTSinger"
mkdir -p "$DATA_ROOT"

if [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON3:-python3}"
fi

echo "Yuaz AI Control Developer Setup"
echo "This downloads training data for developers only. End users will not need this once a pretrained control .pt is bundled."
echo
echo "Dataset preset:"
echo "  1) Chinese Core (recommended): Breathiness + Falsetto + Mixed Voice + Pharyngeal"
echo "  2) Chinese Breathy only (smallest useful AI-control test)"
echo "  3) Chinese Full (also downloads Vibrato/Glissando for future work)"
echo "  4) Use an existing GTSinger folder; do not download"
read -r -p "Choose [1]: " PRESET_CHOICE
PRESET_CHOICE="${PRESET_CHOICE:-1}"
if [ "$PRESET_CHOICE" = "4" ]; then
  read -r -p "Drop the existing GTSinger root here: " EXISTING
  EXISTING="${EXISTING%\"}"; EXISTING="${EXISTING#\"}"; EXISTING="${EXISTING%\'}"; EXISTING="${EXISTING#\'}"
  if [ ! -d "$EXISTING" ]; then echo "Folder not found: $EXISTING"; exit 1; fi
  printf '%s\n' "$EXISTING" > "$DATA_ROOT/ACTIVE_GTSINGER_ROOT.txt"
  echo "Saved developer dataset root: $EXISTING"
  exit 0
fi
case "$PRESET_CHOICE" in
  1) PRESET="chinese-core" ;;
  2) PRESET="chinese-breathy" ;;
  3) PRESET="chinese-full" ;;
  *) echo "Invalid choice"; exit 1 ;;
esac

echo
echo "Download route:"
echo "  1) China mirror hf-mirror.com"
echo "  2) Official Hugging Face (use with VPN)"
read -r -p "Choose [1]: " ROUTE
ROUTE="${ROUTE:-1}"
case "$ROUTE" in
  1) ENDPOINT="https://hf-mirror.com" ;;
  2) ENDPOINT="https://huggingface.co" ;;
  *) echo "Invalid route"; exit 1 ;;
esac

echo
echo "Checking selected files and exact remaining bytes..."
echo "This 0.2.8ai.14 downloader reuses the same manifest/.part files across China and official routes."
"$PYTHON" "$ROOT/scripts/download-gtsinger.py" \
  --preset "$PRESET" --local-dir "$DEST" --endpoint "$ENDPOINT" --dry-run

echo
read -r -p "Start this download? [Y/n]: " CONFIRM
CONFIRM="${CONFIRM:-Y}"
case "$CONFIRM" in
  y|Y|yes|YES) ;;
  *) echo "Cancelled. No dataset files were downloaded by this step."; exit 0 ;;
esac

"$PYTHON" "$ROOT/scripts/download-gtsinger.py" \
  --preset "$PRESET" --local-dir "$DEST" --endpoint "$ENDPOINT"
printf '%s\n' "$DEST" > "$DATA_ROOT/ACTIVE_GTSINGER_ROOT.txt"

echo
echo "Training-data setup complete."
echo "GTSinger root: $DEST"
echo "Re-running this command resumes .part files and skips files whose size already matches the manifest."
echo "Next: ./train-ai-control-foundation.command"
