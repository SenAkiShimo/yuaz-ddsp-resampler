#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -x "$ROOT/.venv/bin/python" ] || { echo "Run setup-macos.command first."; exit 1; }
strip_path() { python3 - "$1" <<'PY'
import shlex,sys
p=shlex.split(sys.argv[1].strip()); print(p[0] if p else '')
PY
}
echo "Drop the prepared voicebank root here, then press Return:"
read -r RAW
BANK="$(strip_path "$RAW")"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$ROOT/.venv/bin/python" - "$BANK" <<'PY'
import json,sys
from pathlib import Path
from yuaz_ddsp_resampler.state import resolve_active_state
bank=Path(sys.argv[1]).expanduser().resolve()
state,info=resolve_active_state(bank,allow_legacy=True,verify=True)
if state is None: raise SystemExit('No valid prepared state found.')
p=state/'highband_profiles_v3.ai14.json'
if not p.exists(): raise SystemExit('highband_profiles_v3.ai14.json not found.')
d=json.loads(p.read_text(encoding='utf-8'))
print('State:',state)
print('Generation:',info.get('generation'))
print('High-Band stats:',json.dumps(d.get('stats',{}),indent=2,ensure_ascii=False))
PY
