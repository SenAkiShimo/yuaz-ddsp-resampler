#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
[ -x .venv/bin/python ] || { echo "Run setup-macos.command first."; exit 1; }
[ -f config.json ] || { echo "Run configure-macos.command first."; exit 1; }
strip_path(){ python3 - "$1" <<'PY'
import shlex,sys;p=shlex.split(sys.argv[1].strip());print(p[0] if p else '')
PY
}
cat <<'TXT'
Yuaz 0.2.8ai.14 — Checkpoint-Isolated Clean Deep

Protection policy:
  • ai.13 OpenUtau resampler remains installed.
  • .yuaz-0.2.8ai13 is read-only and is never cloned into ai.14.
  • ai.14 writes only .yuaz-0.2.8ai14/.staging-* and generations/*.
  • Deep outputs use .ai14 filenames (adapter.ai14.pt, fidelity_refiner.ai14.pt, etc.).
  • Analysis cache uses cache_ai14; high-band cache uses highband_cache_v3_ai14.
  • The selected Yuaz base checkpoint SHA is pinned into base_model.json.

Drop the voicebank root folder here, then press Return:
TXT
read -r RAW; BANK="$(strip_path "$RAW")"; [ -d "$BANK" ] || { echo "Voicebank not found: $BANK"; exit 1; }
"$ROOT/scripts/backup-current-stable.command"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.transaction "$BANK" --project-root "$ROOT" --mode deep
