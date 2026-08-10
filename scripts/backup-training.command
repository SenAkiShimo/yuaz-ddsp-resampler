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
echo "Drop the voicebank root to back up, then press Return:"
read -r RAW
BANK="$(strip_path "$RAW")"
[ -d "$BANK" ] || { echo "Voicebank folder not found: $BANK"; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.backup backup "$BANK" --project-root "$ROOT" --reason manual
