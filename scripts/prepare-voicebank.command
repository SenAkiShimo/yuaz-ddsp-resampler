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
print(parts[0] if parts else '')
PY
}
backup_before_change() {
  local reason="$1"
  echo
  echo "Safety backup before $reason..."
  if ! "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.backup backup "$BANK" --project-root "$ROOT" --reason "$reason"; then
    echo
    echo "ABORTING — previous Yuaz training state was not backed up successfully."
    exit 1
  fi
}

echo "Drop the UTAU/OpenUtau voicebank root folder here, then press Return:"
read -r RAW
BANK="$(strip_path "$RAW")"
if [ ! -d "$BANK" ]; then
  echo "Voicebank folder not found: $BANK"
  exit 1
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
STATE="$BANK/.yuaz-alpha8-rc3-2"

echo
printf '%s\n' \
  "Choose alpha.8 RC3.2 preparation mode:" \
  "  1) Fresh Fast Profile   - profile/canonical/high-band v3/registry only; safety-backup first" \
  "  2) CLEAN DEEP RETRAIN   - backup current/compatible state, then run two-stage training from scratch" \
  "  3) Continue Deep Adapt  - backup current state, continue Stage A and rerun conservative Stage B clarity calibration" \
  "  4) Relearn High-Band    - backup current state, force High-Band v3 source-WAV re-analysis only"
read -r MODE
case "$MODE" in
  1)
    backup_before_change "fast-profile"
    exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.prepare "$BANK" --project-root "$ROOT" --mode profile
    ;;
  2)
    backup_before_change "clean-deep"
    if [ -e "$STATE" ]; then
      rm -rf "$STATE"
      echo "Removed current RC3.2 state only after verified external backup: $STATE"
    fi
    echo "Backup completed successfully; starting fresh two-stage training."
    exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.prepare "$BANK" --project-root "$ROOT" --mode deep
    ;;
  3)
    backup_before_change "continue-deep"
    exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.prepare "$BANK" --project-root "$ROOT" --mode deep
    ;;
  4)
    if [ ! -f "$STATE/manifest.json" ]; then
      echo "No alpha.8 RC3.2 manifest exists yet. Run mode 1 or 2 first."
      exit 1
    fi
    backup_before_change "relearn-highband"
    exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.learn_highband "$BANK" --project-root "$ROOT" --force
    ;;
  *)
    echo "Unknown mode."
    exit 1
    ;;
esac
