#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -f config.json ]; then
  echo "Run scripts/configure-macos.command first."
  exit 1
fi
strip_path() {
  python3 - "$1" <<'PY2'
import shlex, sys
parts = shlex.split(sys.argv[1].strip())
print(parts[0] if parts else "")
PY2
}
echo "Drop one WAV from a prepared multipitch voicebank here, then press Return:"
read -r RAW
WAV="$(strip_path "$RAW")"
if [ ! -f "$WAV" ]; then
  echo "WAV not found: $WAV"
  exit 1
fi
echo "Target tone (for example G4). Press Return for G4:"
read -r TONE
TONE="${TONE:-G4}"
OUT="$ROOT/native-controls-test-output"
rm -rf "$OUT"
mkdir -p "$OUT"
render() {
  local NAME="$1"
  local FLAGS="$2"
  ./yuaz-ddsp-resampler "$WAV" "$OUT/$NAME.wav" "$TONE" 100 "$FLAGS" 0 1200 0 0 100 0 '!120' AA
}
render "00-baseline" ""
render "01-zero-YM0-YD0" "YM0YD0"
render "02-YM-minus100" "YM-100"
render "03-YM-minus50" "YM-50"
render "04-YM-plus50" "YM50"
render "05-YM-plus100" "YM100"
render "06-YD-minus100" "YD-100"
render "07-YD-minus50" "YD-50"
render "08-YD-plus50" "YD50"
render "09-YD-plus100" "YD100"
if cmp -s "$OUT/00-baseline.wav" "$OUT/01-zero-YM0-YD0.wav"; then
  echo "PASS: alpha.8 RC3.2 default and explicit YM0YD0 are byte-identical (both use default YH0; Learned High Band is disabled unless YH is explicitly nonzero)."
else
  echo "WARNING: alpha.8 RC3.2 default and explicit YM0YD0 are not byte-identical."
fi
echo "YM changes learned multipitch timbre routing only; pitch stays at $TONE."
echo "YD changes learned detail contribution across Adapter, Fidelity Refiner, and source high band."
echo "YH is 0..100 restoration amount; YH100 performs full post-12-kHz restoration."
open "$OUT" 2>/dev/null || true
