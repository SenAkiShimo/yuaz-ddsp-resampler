#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -x .venv/bin/python ]; then
  echo "Run scripts/setup-macos.command first."
  exit 1
fi
if [ ! -f config.json ]; then
  echo "Run scripts/configure-macos.command first."
  exit 1
fi
strip_path() {
  python3 - "$1" <<'PY2'
import shlex, sys
parts=shlex.split(sys.argv[1].strip())
print(parts[0] if parts else '')
PY2
}
echo "Drop the alpha.8 prepared voicebank root here, then press Return:"
read -r RAW
BANK="$(strip_path "$RAW")"
if [ ! -f "$BANK/.yuaz-alpha8-rc3-2/manifest.json" ]; then
  echo "No .yuaz-alpha8-rc3-2/manifest.json found. Run Prepare Voicebank first."
  exit 1
fi
echo
echo "Relearn High-Band from source WAVs?"
echo "  1) Force re-analysis (recommended when old high-band state is suspect)"
echo "  2) Reuse valid alpha.8 high-band cache"
read -r MODE
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [ "$MODE" = "1" ]; then
  exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.learn_highband "$BANK" --project-root "$ROOT" --force
else
  exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.learn_highband "$BANK" --project-root "$ROOT"
fi
