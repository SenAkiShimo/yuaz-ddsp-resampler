#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -x .venv/bin/python ] || { echo "Run setup-macos.command first."; exit 1; }
[ -f config.json ] || { echo "Run configure-macos.command first."; exit 1; }
strip_path() { python3 - "$1" <<'PY'
import shlex,sys
parts=shlex.split(sys.argv[1].strip()); print(parts[0] if parts else '')
PY
}
echo "Drop the UTAU/OpenUtau voicebank root folder here, then press Return:"
read -r RAW
BANK="$(strip_path "$RAW")"
[ -d "$BANK" ] || { echo "Voicebank folder not found: $BANK"; exit 1; }
echo
printf '%s\n' \
  "Choose 0.2.8ai.11 preparation mode (stable RC4.2 is never modified):" \
  "  1) SAFE AI PRODUCTION CLEAN DEEP (recommended) - fresh analysis into .yuaz-0.2.8ai11" \
  "  2) Snapshot current stable baseline into AI namespace - no retrain" \
  "  3) Continue AI Deep - isolated copy of current AI generation" \
  "  4) Relearn High-Band v3 in AI namespace only" \
  "Press Return for option 1."
read -r MODE
MODE="${MODE:-1}"
case "$MODE" in
  1) TX=deep;; 2) TX=adopt;; 3) TX=continue;; 4) TX=highband;; *) echo "Unknown mode."; exit 1;;
esac
"$ROOT/scripts/backup-current-stable.command"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.transaction "$BANK" --project-root "$ROOT" --mode "$TX"
