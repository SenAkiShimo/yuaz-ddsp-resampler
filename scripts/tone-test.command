#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -f config.json ]; then
  echo "Run scripts/configure-macos.command first."
  exit 1
fi
strip_path() {
  python3 - "$1" <<'PY'
import shlex, sys
parts = shlex.split(sys.argv[1].strip())
print(parts[0] if parts else "")
PY
}
echo "Drop a voiced UTAU sample WAV here, then press Return:"
read -r RAW
WAV="$(strip_path "$RAW")"
if [ ! -f "$WAV" ]; then
  echo "WAV not found: $WAV"
  exit 1
fi
OUT="$ROOT/tone-test-output"
rm -rf "$OUT"
mkdir -p "$OUT"
for TONE in C4 G4 C5; do
  ./yuaz-ddsp-resampler "$WAV" "$OUT/${TONE}.wav" "$TONE" 100 "" 0 1000 0 0 100 0 '!120' AA
done
open "$OUT"
