#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -x .venv/bin/python ] || { echo "Run setup-macos.command first."; exit 1; }
[ -f config.json ] || { echo "Run configure-macos.command first."; exit 1; }
strip_path() { python3 - "$1" <<'PY'
import shlex,sys
p=shlex.split(sys.argv[1].strip()); print(p[0] if p else '')
PY
}
echo "Drop the voicebank root to roll back, then press Return:"
read -r RAW
BANK="$(strip_path "$RAW")"
[ -d "$BANK" ] || { echo "Voicebank folder not found: $BANK"; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$ROOT/.venv/bin/python" - "$BANK" "$ROOT" <<'PY'
import json,sys,time
from pathlib import Path
from yuaz_ddsp_resampler.client import ENGINE_VERSION,ping,send
from yuaz_ddsp_resampler.state import rollback_to_previous,write_local_registry,merge_global_registry
bank=Path(sys.argv[1]).resolve(); root=Path(sys.argv[2]).resolve()
cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
host=cfg.get('host','127.0.0.1'); port=int(cfg.get('port',47886)); rid=cfg.get('runtime_id')
st=ping(host,port)
if st and st.get('ready'):
    if st.get('engine_version')!=ENGINE_VERSION or st.get('runtime_id')!=rid:
        raise SystemExit('Another Yuaz runtime occupies the 0.2.8ai.14 port; rollback refused.')
    if int(st.get('active_renders') or 0)>0:
        raise SystemExit('OpenUtau is actively rendering. Rollback refused; try again after rendering stops.')
    try: send(host,port,{'action':'shutdown','runtime_id':rid},timeout=2)
    except Exception: pass
    time.sleep(0.3)
target,payload=rollback_to_previous(bank)
print('ACTIVE rolled back to:',target)
print('Former generation retained as previous:',payload.get('previous_generation'))
try:
    reg=write_local_registry(bank,target)
    merge_global_registry(Path(cfg['registry_path']),reg)
except Exception as exc:
    print('WARNING: registry accelerator refresh failed:',exc)
    print('The local ACTIVE generation remains authoritative.')
PY
