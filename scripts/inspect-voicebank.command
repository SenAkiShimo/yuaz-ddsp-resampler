#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -x "$ROOT/.venv/bin/python" ] || { echo "Run setup-macos.command first."; exit 1; }
strip_path() { python3 - "$1" <<'PY'
import shlex,sys
p=shlex.split(sys.argv[1].strip()); print(p[0] if p else '')
PY
}
echo "Drop a prepared voicebank root folder here, then press Return:"
read -r RAW
BANK="$(strip_path "$RAW")"
[ -d "$BANK" ] || { echo "Voicebank folder not found: $BANK"; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/.venv/bin/python" - "$BANK" <<'PY'
import json,sys
from pathlib import Path
from yuaz_ddsp_resampler.state import resolve_active_state, container_dir, FINGERPRINT_FILE
bank=Path(sys.argv[1]).expanduser().resolve()
state,info=resolve_active_state(bank,allow_legacy=True,verify=True)
print('Voicebank:',bank)
print('State source:',info.get('source'))
print('Active generation:',info.get('generation'))
if state is None:
    print('No valid RC3.3/RC3.2 prepared state found.')
    raise SystemExit(1)
print('Resolved state:',state)
print('Pinned fingerprint:', 'yes' if (state/FINGERPRINT_FILE).is_file() else 'legacy/unpinned')
print()
def load(name):
    p=state/name
    if not p.exists(): return None
    return json.loads(p.read_text(encoding='utf-8'))
sub=load('subbanks.json') or {}
labels=[f"{x.get('label')}@{x.get('anchor_note')}" for x in sub.get('subbanks',[])]
print('Routing summary:')
print('  strategy:',sub.get('strategy'))
print('  prototype_count:',sub.get('prototype_count'))
print('  prototypes:',', '.join(labels))
print('  prefix_map_authoritative:',sub.get('prefix_map_authoritative'))
print()
art=load('articulation/index.json') or {}
print('Canonical articulation:')
print('  strategy:',art.get('strategy'))
print('  alias_count:',art.get('alias_count'))
print('  multipitch_canonical_count:',art.get('multipitch_canonical_count'))
print('  single_neutral_fallback_count:',art.get('single_neutral_fallback_count'))
print('  mean_coherence:',art.get('mean_coherence'))
print()
hb=load('highband_profiles_v3.ai14.json') or {}
print('Learned High-Band v3:')
print('  present:',bool(hb))
print('  stats:',hb.get('stats'))
print()
profile=load('profile.json') or {}
training=load('training.ai14.json')
fidelity=load('fidelity_training.ai14.json')
clarity=load('clarity_calibration.ai14.json')
deep=load('deep_validation.ai14.json')
print('Training:')
print('  adapter:', 'yes' if (state/'adapter.ai14.pt').is_file() else 'no')
print('  timbre profiles:', 'yes' if (state/'timbre_profiles.ai14.pt').is_file() else 'no')
print('  clarity calibration:', 'yes' if clarity else 'no')
print('  fidelity refiner:', 'yes' if (state/'fidelity_refiner.ai14.pt').is_file() else 'no')
print('  fidelity metadata:', 'yes' if fidelity else 'no')
print('  fidelity stage:', (fidelity or {}).get('stage'))
print('  fidelity hard limit:', (fidelity or {}).get('residual_rms_hard_limit'))
print('  deep validation:', 'yes' if deep else 'legacy/not present')
if deep:
    print('  activation_safe:', deep.get('activation_safe'))
    print('  stage A selected:', deep.get('stage_a_selected_checkpoint'))
    print('  stage A safe fallback:', deep.get('stage_a_safe_fallback'))
    print('  stage C accepted:', deep.get('stage_c_accepted'))
print('  cache files:', len(list((state/'cache').glob('*.npz'))) if (state/'cache').exists() else 0)
print()
for name,data in [('profile.json',profile),('training.ai14.json',training),('clarity_calibration.ai14.json',clarity),('fidelity_training.ai14.json',fidelity),('deep_validation.ai14.json',deep)]:
    print(f'--- {name} ---')
    print(json.dumps(data,indent=2,ensure_ascii=False) if data is not None else 'not found')
PY
