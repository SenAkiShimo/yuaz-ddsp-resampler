#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -f config.json ] || { echo "Run scripts/configure-macos.command first."; exit 1; }

strip_path() {
  python3 - "$1" <<'PY'
import shlex,sys
parts=shlex.split(sys.argv[1].strip())
print(parts[0] if parts else "")
PY
}

WAV="${1:-}"
if [ -z "$WAV" ]; then
  echo "Drop one WAV from a prepared voicebank here, then press Return:"
  read -r RAW
  WAV="$(strip_path "$RAW")"
fi
[ -f "$WAV" ] || { echo "WAV not found: $WAV"; exit 1; }

TONE="${2:-G4}"
OUT="$ROOT/yf-yb-listen-test-output"
rm -rf "$OUT"
mkdir -p "$OUT"

render() {
  ./yuaz-ddsp-resampler "$WAV" "$OUT/$1.wav" "$TONE" 100 "$2" 0 3000 0 0 100 0 '!120' AA
}

render "00-baseline" ""
render "01-yb50" "YB50"
render "02-yb100" "YB100"
render "03-yf50" "YF50"
render "04-yf100" "YF100"
render "05-yb50-yf50" "YB50YF50"

echo "Outputs: $OUT"
open "$OUT" 2>/dev/null || true
