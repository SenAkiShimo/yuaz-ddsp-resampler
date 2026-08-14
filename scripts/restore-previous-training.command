#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -x .venv/bin/python ] || { echo "Run setup-macos.command first."; exit 1; }
strip_path() { python3 - "$1" <<'PY'
import shlex,sys
p=shlex.split(sys.argv[1].strip()); print(p[0] if p else '')
PY
}
echo "Drop the voicebank root, then press Return:"
read -r RAW
BANK="$(strip_path "$RAW")"
[ -d "$BANK" ] || { echo "Voicebank folder not found: $BANK"; exit 1; }
echo "Restore from the latest external backup:"
echo "  1) 0.2.8ai.14 state (restores only the current namespace)"
echo "  2) RC3.2 legacy baseline"
echo "Stable RC4.2 is intentionally not modified by this AI restore tool."
read -r MODE
case "$MODE" in 1) TARGET=ai;; 2) TARGET=rc3-2;; *) echo "Unknown mode."; exit 1;; esac
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.backup restore "$BANK" --target "$TARGET"
