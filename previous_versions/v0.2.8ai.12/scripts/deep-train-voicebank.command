#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -x .venv/bin/python ] || { echo "Run setup-macos.command first."; exit 1; }
[ -f config.json ] || { echo "Run configure-macos.command first."; exit 1; }
strip_path() { python3 - "$1" <<'PY2'
import shlex,sys
parts=shlex.split(sys.argv[1].strip()); print(parts[0] if parts else '')
PY2
}
cat <<'TXT'
Yuaz 0.2.8ai.12 — SAFE Production Clean Deep

Safety policy:
  1) 0.2.7 AI.3 and RC4.2 engines/wrappers stay installed in OpenUtau.
  2) Voicebank states .yuaz-alpha8-rc4-3-ai3 and .yuaz-alpha8-rc3-3 are read-only.
  3) External full-container backups are made before training.
  4) Deep writes only .yuaz-0.2.8ai12/.staging-*.
  5) ACTIVE switches only inside .yuaz-0.2.8ai12 after validation.
  6) The existing four-axis AI.3 foundation can be reused by copy; its source remains untouched.

Drop the UTAU/OpenUtau voicebank root folder here, then press Return:
TXT
read -r RAW
BANK="$(strip_path "$RAW")"
[ -d "$BANK" ] || { echo "Voicebank folder not found: $BANK"; exit 1; }
"$ROOT/scripts/backup-current-stable.command"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.transaction "$BANK" --project-root "$ROOT" --mode deep
