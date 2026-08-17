#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || exit 1
WAV="${1:-}"
if [ -z "$WAV" ]; then
  read -r WAV
fi
[ -f "$WAV" ] || exit 1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$ROOT" "$WAV" <<'PY'
import json,sys,tempfile
from pathlib import Path
from yuaz_ddsp_resampler import client
root=Path(sys.argv[1]).resolve()
wav=Path(sys.argv[2]).resolve()
config,config_path=client.load_config(root)
host=config.get('host','127.0.0.1')
port=int(config.get('port',client.DEFAULT_PORT))
runtime_id=str(config.get('runtime_id') or client.ENGINE_VERSION)
status=client.ping(host,port)
if not client._status_matches(status,runtime_id,root):
    client.start_server(root,config_path,host,port,runtime_id)
for value in (-100,0,100):
    with tempfile.NamedTemporaryFile(suffix='.wav',delete=False) as f:
        out=f.name
    request={
        'input':str(wav),'output':out,'tone':'C4','velocity':100.0,
        'flags':f'YT{value}','offset':0.0,'length':1000.0,
        'consonant':0.0,'cutoff':0.0,'volume':100.0,'modulation':0.0,
        'tempo':'!120','pitch':'AA',
    }
    result=client.send(host,port,{'action':'render','request':request,'runtime_id':runtime_id},timeout=600)
    Path(out).unlink(missing_ok=True)
    if not result.get('ok'):
        raise RuntimeError(result.get('error','render failed'))
    effect=None
    for item in result.get('yuaz_ai_effects') or []:
        if 'tension' in (item.get('pack_controls') or []):
            effect=item
            break
    payload={
        'yt':value,
        'backend':result.get('yuaz_tension_backend'),
        'packs':result.get('yuaz_ai_direct_controls'),
        'loaded':effect is not None,
    }
    if effect is not None:
        payload.update({
            'raw_s':effect.get('raw_spectral_rms',0.0),
            'raw_ap':effect.get('raw_ap_rms',0.0),
            'raw_gate':effect.get('raw_gate_rms',0.0),
            'applied_s':effect.get('applied_spectral_log_rms',0.0),
            'applied_ap':effect.get('applied_ap_rms',0.0),
            'applied_gate':effect.get('applied_gate_rms',0.0),
            'gain':effect.get('runtime_gain',0.0),
            'collapsed':effect.get('collapsed',False),
        })
    print(json.dumps(payload,separators=(',',':')))
PY