#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || exit 1
BANK="${1:-}"
if [ -z "$BANK" ]; then
  read -r BANK
fi
[ -d "$BANK" ] || exit 1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$ROOT" "$BANK" <<'PY'
import json,sys
from pathlib import Path
from yuaz_ddsp_resampler.checkpoint_identity import checkpoint_identity_sha
from yuaz_ddsp_resampler.transaction import (
    _attach_ai_control_foundation,
    _attach_ai_phonation_foundation,
    _quiesce_runtime,
    _safe_remove_staging,
    _state_base_sha,
    _write_base_model_metadata,
    load_config,
)
from yuaz_ddsp_resampler.state import (
    begin_generation,
    clone_state,
    commit_generation,
    merge_global_registry,
    resolve_active_state,
    write_local_registry,
)
root=Path(sys.argv[1]).resolve()
bank=Path(sys.argv[2]).expanduser().resolve()
config=load_config(root)
source,_=resolve_active_state(bank,verify=True)
if source is None:
    raise RuntimeError('No ai.14 state')
current_sha=str(config.get('base_checkpoint_sha256') or checkpoint_identity_sha(config['checkpoint']))
source_sha=_state_base_sha(source)
print(json.dumps({'type':'source','generation':source.name,'source_checkpoint_sha256':source_sha,'current_checkpoint_sha256':current_sha,'compatible':source_sha==current_sha},separators=(',',':')))
if source_sha!=current_sha:
    raise RuntimeError('Active ai.14 state uses another base checkpoint; refusing control-only repin')
_quiesce_runtime(config,'current control pack repin')
generation,staging=begin_generation(bank,'control-repin')
try:
    clone_state(source,staging,link_caches=True)
    tech=_attach_ai_control_foundation(root,staging,source_state=None)
    if tech is None:
        meta=json.loads((staging/'ai_control_training.ai14.json').read_text(encoding='utf-8'))
        raise RuntimeError('Technique foundation rejected: '+str(meta.get('reason') or 'unknown'))
    phon=_attach_ai_phonation_foundation(root,staging,source_state=None)
    if phon is None:
        meta=json.loads((staging/'ai_phonation_training.ai14.json').read_text(encoding='utf-8'))
        raise RuntimeError('Phonation foundation rejected: '+str(meta.get('reason') or 'unknown'))
    _write_base_model_metadata(staging,config)
    final,_=commit_generation(
        bank,generation,staging,
        reason='0.2.8ai.14-current-control-pack-repin',
        acoustic_base='0.2.8ai.14-checkpoint-isolated-twelve-control-ddsp',
    )
    payload=write_local_registry(bank,final)
    registry=Path(config['registry_path']).expanduser().resolve()
    try:
        merge_global_registry(registry,payload)
    except Exception:
        pass
    print(json.dumps({'type':'committed','generation':final.name,'technique':str(tech.name),'phonation':str(phon.name)},separators=(',',':')))
    print(final)
except Exception:
    _safe_remove_staging(staging)
    raise
PY
