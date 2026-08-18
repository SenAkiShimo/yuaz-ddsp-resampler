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
if _state_base_sha(source)!=current_sha:
    raise RuntimeError('Active ai.14 state uses another base checkpoint')
_quiesce_runtime(config,'technique repin')
generation,staging=begin_generation(bank,'technique-repin')
try:
    clone_state(source,staging,link_caches=True)
    target=_attach_ai_control_foundation(root,staging,source_state=None)
    if target is None:
        meta=json.loads((staging/'ai_control_training.ai14.json').read_text(encoding='utf-8'))
        raise RuntimeError(str(meta.get('reason') or 'technique foundation rejected'))
    _write_base_model_metadata(staging,config)
    final,_=commit_generation(
        bank,generation,staging,
        reason='0.2.8ai.14-technique-repin',
        acoustic_base='0.2.8ai.14-checkpoint-isolated-twelve-control-ddsp',
    )
    payload=write_local_registry(bank,final)
    registry=Path(config['registry_path']).expanduser().resolve()
    try:
        merge_global_registry(registry,payload)
    except Exception:
        pass
    print(final)
except Exception:
    _safe_remove_staging(staging)
    raise
PY
