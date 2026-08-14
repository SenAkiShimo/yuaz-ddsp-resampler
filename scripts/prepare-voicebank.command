#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
[ -x .venv/bin/python ] || { echo "Run setup-macos.command first."; exit 1; }
[ -f config.json ] || { echo "Run configure-macos.command first."; exit 1; }
strip_path(){ python3 - "$1" <<'PY'
import shlex,sys;p=shlex.split(sys.argv[1].strip());print(p[0] if p else '')
PY
}
echo "Drop the voicebank root folder here, then press Return:"; read -r RAW; BANK="$(strip_path "$RAW")"
[ -d "$BANK" ] || { echo "Voicebank not found: $BANK"; exit 1; }
printf '%s\n' "Choose ai.14 mode:" "  1) Clean Deep (recommended; never reads ai.13 learned state)" "  2) Continue matching ai.14 generation" "  3) Relearn High-Band in matching ai.14 generation" "Press Return for 1."
read -r M; M="${M:-1}"; case "$M" in 1) TX=deep;;2) TX=continue;;3) TX=highband;;*) exit 1;; esac
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.transaction "$BANK" --project-root "$ROOT" --mode "$TX"
