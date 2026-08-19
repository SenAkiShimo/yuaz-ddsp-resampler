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
import json,sys,shutil,time
from pathlib import Path
from yuaz_ddsp_resampler.ai_vocal_controls import load_ai_control_adapter
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
    atomic_write_json,
    begin_generation,
    clone_state,
    commit_generation,
    merge_global_registry,
    resolve_active_state,
    sha256,
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


def phonation_candidates():
    home=Path.home()
    out=[root/'control_models'/'ai_phonation_foundation-v1.pt']
    out.extend(sorted(root.parent.glob('yuaz-ddsp-resampler-v0.2.8ai.14*/control_models/ai_phonation_foundation-v1.pt')))
    out.extend([
        home/'Documents'/'Yuaz-DDSP-Backups'/'control-models'/'ai_phonation_foundation-v1-PhonationModes-MOCHA.pt',
        home/'Documents'/'Yuaz-DDSP-Backups'/'control-models'/'ai_phonation_foundation-v1-VQS-MOCHA.pt',
        home/'Library'/'Application Support'/'YuazDDSP'/'0.2.8ai.14'/'control_models'/'ai_phonation_foundation-v1.pt',
        home/'Downloads'/'yuaz-ddsp-resampler-v0.2.8ai.14-phonation-fix'/'control_models'/'ai_phonation_foundation-v1.pt',
    ])
    seen=set()
    for p in out:
        p=Path(p).expanduser()
        try: key=str(p.resolve())
        except Exception: key=str(p)
        if key in seen: continue
        seen.add(key)
        if p.is_file():
            yield p.resolve()


def select_phonation():
    expected_controls=('tension','voicing')
    expected_modes=('signed','signed')
    expected_scopes=('spectral','ap','gate')
    checked=0
    for p in phonation_candidates():
        checked+=1
        row={'type':'phonation_candidate','path':str(p)}
        try:
            pack,meta=load_ai_control_adapter(p,device='cpu',expected_controls=expected_controls)
            controls=tuple(pack.control_names)
            modes=tuple(pack.control_modes)
            scopes=tuple(pack.output_scopes)
            backend=str(meta.get('feature_backend') or '')
            model_sha=str(meta.get('checkpoint_sha256') or '')
            ok=(controls==expected_controls and modes==expected_modes and scopes==expected_scopes and backend=='yuaz-native-ddsp-v1' and model_sha==current_sha)
            row.update({'controls':list(controls),'modes':list(modes),'scopes':list(scopes),'feature_backend':backend,'checkpoint_sha256':model_sha,'compatible':ok})
            print(json.dumps(row,separators=(',',':')))
            if ok:
                return p,meta
        except Exception as exc:
            row.update({'compatible':False,'error':str(exc)})
            print(json.dumps(row,separators=(',',':')))
    raise RuntimeError(f'No compatible phonation foundation found after checking {checked} candidate(s)')


def pin_phonation(staging):
    foundation,meta=select_phonation()
    target=Path(staging)/'ai_phonation_adapter.ai14.pt'
    shutil.copy2(foundation,target)
    atomic_write_json(Path(staging)/'ai_phonation_training.ai14.json',{
        'format':1,
        'accepted':True,
        'backend':'ai-ddsp',
        'controls':['tension','voicing'],
        'source':str(foundation),
        'foundation_sha256':sha256(foundation),
        'foundation_metadata':meta,
        'frozen_during_voicebank_deep':True,
        'created_at':time.time(),
    })
    print('AI Phonation Foundation selected:', foundation)
    print('Pinned phonation copy:', target)
    return target

_quiesce_runtime(config,'current control pack repin')
generation,staging=begin_generation(bank,'control-repin')
try:
    clone_state(source,staging,link_caches=True)
    tech=_attach_ai_control_foundation(root,staging,source_state=None)
    if tech is None:
        meta=json.loads((staging/'ai_control_training.ai14.json').read_text(encoding='utf-8'))
        raise RuntimeError('Technique foundation rejected: '+str(meta.get('reason') or 'unknown'))
    phon=pin_phonation(staging)
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
