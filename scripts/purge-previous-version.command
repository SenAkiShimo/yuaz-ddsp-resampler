#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HOME/Library/OpenUtau/Resamplers"
PREVIOUS=(
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.6-alpha.2"
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.6-alpha.1"
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.5-alpha.2"
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.5-alpha.1"
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.4-alpha.2"
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.4-alpha.1"
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.3-alpha.1"
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.2-alpha.1"
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.1-alpha.1"
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.0-alpha.1"
)

pkill -f 'yuaz-ddsp-resampler-v0\.2\.6-alpha\.2' 2>/dev/null || true
pkill -f 'yuaz-ddsp-resampler-v0\.2\.6-alpha\.1' 2>/dev/null || true
pkill -f 'yuaz-ddsp-resampler-v0\.2\.5-alpha\.2' 2>/dev/null || true
pkill -f 'yuaz-ddsp-resampler-v0\.2\.5-alpha\.1' 2>/dev/null || true
pkill -f 'yuaz-ddsp-resampler-v0\.2\.4-alpha\.2' 2>/dev/null || true
pkill -f 'yuaz-ddsp-resampler-v0\.2\.4-alpha\.1' 2>/dev/null || true
pkill -f 'yuaz-ddsp-resampler-v0\.2\.3-alpha\.1' 2>/dev/null || true
pkill -f 'yuaz-ddsp-resampler-v0\.2\.2-alpha\.1' 2>/dev/null || true
rm -f \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.6-alpha.2.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.6-alpha.1.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.5-alpha.2.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.5-alpha.1.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.4-alpha.2.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.4-alpha.1.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.3-alpha.1.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.2-alpha.1.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.1-alpha.1.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.0-alpha.1.sh"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  for OLD in "${PREVIOUS[@]}"; do
    if [ -x "$OLD/.venv/bin/python" ]; then
      mv "$OLD/.venv" "$ROOT/.venv"
      echo "Moved Python environment from $(basename "$OLD")."
      break
    fi
  done
fi

for OLD in "${PREVIOUS[@]}"; do
  if [ -d "$OLD" ] && [ "$(cd "$OLD" 2>/dev/null && pwd || true)" != "$ROOT" ]; then
    rm -rf "$OLD"
    echo "Removed previous program folder: $(basename "$OLD")"
  fi
done

rm -f "$ROOT/.engine-start.lock"
echo "Previous program versions removed. Voicebank .yuaz data and Yuaz SGR were preserved."
