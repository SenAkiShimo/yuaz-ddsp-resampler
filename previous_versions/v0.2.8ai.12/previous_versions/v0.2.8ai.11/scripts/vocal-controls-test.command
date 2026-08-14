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
echo "Drop one WAV from a prepared voicebank here, then press Return:"
read -r RAW
WAV="$(strip_path "$RAW")"
if [ ! -f "$WAV" ]; then
  echo "WAV not found: $WAV"
  exit 1
fi
echo "Target tone (default G4):"
read -r TONE
TONE="${TONE:-G4}"
OUT="$ROOT/vocal-controls-test-output"
rm -rf "$OUT" && mkdir -p "$OUT"
render() {
  local NAME="$1" FLAGS="$2"
  local START END ELAPSED
  START=$(python3 - <<'PY'
import time
print(time.perf_counter())
PY
)
  ./yuaz-ddsp-resampler "$WAV" "$OUT/$NAME.wav" "$TONE" 100 "$FLAGS" 0 3000 0 0 100 0 '!120' AA
  END=$(python3 - <<'PY'
import time
print(time.perf_counter())
PY
)
  ELAPSED=$(python3 - "$START" "$END" <<'PY'
import sys
print(float(sys.argv[2])-float(sys.argv[1]))
PY
)
  python3 - "$NAME" "$FLAGS" "$ELAPSED" <<'PY'
import sys
elapsed=float(sys.argv[3]); target=3.0
print(f"{sys.argv[1]:24s} flags={sys.argv[2] or '(none)':28s} wall={elapsed:.4f}s  wall_RTF={elapsed/target:.4f}")
PY
}
render "00-baseline" ""
render "01-tension-minus100" "YT-100"
render "02-tension-plus100" "YT100"
render "03-breathiness-minus100" "YB-100"
render "04-breathiness-plus100" "YB100"
render "05-voicing-minus100" "YV-100"
render "06-voicing-plus100" "YV100"
render "07-formant-minus100" "YG-100"
render "08-formant-plus100" "YG100"
render "09-mouth-closed100" "YO-100"
render "10-mouth-open100" "YO100"
render "11-falsetto-plus100" "YF100"
render "12-mixed-plus100" "YX100"
render "13-pharyngeal-plus100" "YP100"
render "14-ai-techniques" "YB70YF70YX40YP40"
render "15-all-controls" "YM15YD10YH100YT35YB25YV20YG-25YO30YF35YX25YP20"
echo
echo "Outputs: $OUT"
open "$OUT" 2>/dev/null || true
