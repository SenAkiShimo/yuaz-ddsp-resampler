#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -x .venv/bin/python ]; then echo "Run scripts/setup-macos.command first."; exit 1; fi
strip_path() { python3 - "$1" <<'PY'
import shlex,sys
p=shlex.split(sys.argv[1].strip()); print(p[0] if p else '')
PY
}
echo "Drop the voicebank root, then press Return:"
read -r RAW
BANK="$(strip_path "$RAW")"
[ -d "$BANK" ] || { echo "Voicebank folder not found: $BANK"; exit 1; }
echo "Restore which state from the latest external backup?"
echo "  1) Previous compatible state"
echo "  2) RC3.2 state"
read -r MODE
case "$MODE" in 1) TARGET=rc3-1;; 2) TARGET=rc3-2;; *) echo "Unknown mode."; exit 1;; esac
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.backup restore "$BANK" --target "$TARGET"
