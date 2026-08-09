#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HOME/Library/OpenUtau/Resamplers"
rm -f \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.7-alpha.1.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.6-alpha.2.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.6-alpha.1.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.5-alpha.2.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.5-alpha.1.sh" \
  "$DEST/Yuaz-DDSP-Resampler" \
  "$DEST/Yuaz-DDSP-Resampler.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2-alpha.1.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.0-alpha.1.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.1-alpha.1.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.2-alpha.1.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.3-alpha.1.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.4-alpha.1.sh" \
  "$DEST/Yuaz-DDSP-Resampler-v0.2.4-alpha.2.sh"
for NAME in Yuaz-DDSP-Stage3 Yuaz-DDSP-Stage3.sh Yuaz-DDSP-Stage3.1 Yuaz-DDSP-Stage3.1.sh Yuaz-DDSP-Stage3.2 Yuaz-DDSP-Stage3.2.sh Yuaz-DDSP-Stage3.3 Yuaz-DDSP-Stage3.3.sh Yuaz-DDSP-Stage3.4 Yuaz-DDSP-Stage3.4.sh; do
  rm -f "$DEST/$NAME"
done
pkill -f 'yuaz-ddsp-resampler-v0\.2\.7-alpha\.1' 2>/dev/null || true
pkill -f 'yuaz-ddsp-resampler-v0\.2\.6-alpha\.2' 2>/dev/null || true
pkill -f 'yuaz-ddsp-resampler-v0\.2\.6-alpha\.1' 2>/dev/null || true
pkill -f 'yuaz-ddsp-resampler-v0\.2\.5-alpha\.2' 2>/dev/null || true
pkill -f 'yuaz-ddsp-resampler-v0\.2\.5-alpha\.1' 2>/dev/null || true
pkill -f 'yuaz-ddsp-resampler-v0\.2\.4-alpha\.2' 2>/dev/null || true
pkill -f 'yuaz-ddsp-resampler-v0\.2\.4-alpha\.1' 2>/dev/null || true
pkill -f 'yuaz-ddsp-resampler-v0\.2\.3-alpha\.1' 2>/dev/null || true
pkill -f "$ROOT/config.json" 2>/dev/null || true
rm -f "$ROOT/.engine-start.lock"
echo "Removed installed Yuaz DDSP Resampler entries from OpenUtau."
