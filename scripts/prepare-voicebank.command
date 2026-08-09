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
  python3 - "$1" <<'PY'
import shlex, sys
parts = shlex.split(sys.argv[1].strip())
print(parts[0] if parts else "")
PY
}
echo "Drop the UTAU/OpenUtau voicebank root folder here, then press Return:"
read -r RAW
BANK="$(strip_path "$RAW")"
if [ ! -d "$BANK" ]; then
  echo "Voicebank folder not found: $BANK"
  exit 1
fi
echo
printf '%s\n' \
  "Choose preparation mode:" \
  "  1) Fast Profile - build canonical articulation + refresh loudness/registry; no gradient training" \
  "  2) Quick Adapt  - canonical articulation + existing adaptation + small fidelity update" \
  "  3) Deep Adapt   - full acoustic retraining; not needed when upgrading an already-prepared bank"
read -r MODE
case "$MODE" in
  1) MODE_NAME=profile ;;
  3) MODE_NAME=deep ;;
  *) MODE_NAME=quick ;;
esac
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.prepare "$BANK" --project-root "$ROOT" --mode "$MODE_NAME"
