#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; YUAZ="${YUAZ_SGR_ROOT:-$HOME/Downloads/yuaz-sgr}"
echo "Drop a Yuaz checkpoint here, then press Return:"; read -r RAW
P="$(python3 - "$RAW" <<'PY'
import shlex,sys;p=shlex.split(sys.argv[1]);print(p[0] if p else '')
PY
)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"; .venv/bin/python -m yuaz_ddsp_resampler.checkpoint_registry import "$P" --yuaz-repo "$YUAZ"
echo "Import complete. Run select-yuaz-checkpoint.command to activate it for ai.14."
