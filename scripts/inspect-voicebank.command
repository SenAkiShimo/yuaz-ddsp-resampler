#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
strip_path() {
  python3 - "$1" <<'PY'
import shlex, sys
parts = shlex.split(sys.argv[1].strip())
print(parts[0] if parts else "")
PY
}
echo "Drop a prepared voicebank root folder here, then press Return:"
read -r RAW
BANK="$(strip_path "$RAW")"
python3 - "$BANK" <<'PY'
import json, sys
from pathlib import Path
bank=Path(sys.argv[1])
y=bank/'.yuaz-alpha8-rc3-2'
sub=y/'subbanks.json'
if sub.exists():
    data=json.loads(sub.read_text(encoding='utf-8'))
    labels=[f"{x.get('label')}@{x.get('anchor_note')}" for x in data.get('subbanks', [])]
    print('Routing summary:')
    print('  strategy:', data.get('strategy'))
    print('  prototype_count:', data.get('prototype_count'))
    print('  prototypes:', ', '.join(labels))
    print('  prefix_map_authoritative:', data.get('prefix_map_authoritative'))
    print('  fallback_created_prototypes:', data.get('fallback_created_prototypes'))
    print('  fallback_assignment_count:', data.get('fallback_assignment_count'))
    print()
loudness_path=y/'loudness.json'
if loudness_path.exists():
    loudness=json.loads(loudness_path.read_text(encoding='utf-8'))
    print('Loudness normalization:')
    print('  enabled:', loudness.get('enabled'))
    print('  mode:', loudness.get('mode'))
    print('  target_active_rms_dbfs:', loudness.get('target_active_rms_dbfs'))
    print('  tolerance_db:', loudness.get('tolerance_db'))
    print('  peak_ceiling_dbfs:', loudness.get('peak_ceiling_dbfs'))
    print('  source_measured_median_dbfs:', loudness.get('source_measured_median_dbfs'))
    print('  source_quietest_dbfs:', loudness.get('source_quietest_dbfs'))
    print('  source_loudest_dbfs:', loudness.get('source_loudest_dbfs'))
    for label, stats in (loudness.get('subbanks') or {}).items():
        print(f"  subbank {label}: {stats.get('median_active_rms_dbfs')} dBFS ({stats.get('count')} entries)")
    print()
articulation_index=y/'articulation'/'index.json'
if articulation_index.exists():
    art=json.loads(articulation_index.read_text(encoding='utf-8'))
    print('Canonical articulation:')
    print('  strategy:', art.get('strategy'))
    print('  alias_count:', art.get('alias_count'))
    print('  multipitch_canonical_count:', art.get('multipitch_canonical_count'))
    print('  single_neutral_fallback_count:', art.get('single_neutral_fallback_count'))
    print('  mean_coherence:', art.get('mean_coherence'))
    print('  clarity_guard:', art.get('clarity_guard'))
    print()

profile_path=y/'profile.json'
if profile_path.exists():
    profile=json.loads(profile_path.read_text(encoding='utf-8'))
    print('Articulation summary:')
    print('  training_version:', profile.get('articulation_training_version'))
    print('  median_voiced_span_ms:', profile.get('median_articulation_voiced_span_ms'))
    print('  median_confidence:', profile.get('median_articulation_confidence'))
    print('  cache_format:', profile.get('cache_format'))
    print()
for name in ('subbanks.json','loudness.json','profile.json','training.json','clarity_calibration.json','fidelity_training.json'):
    p=y/name
    print(f'--- {name} ---')
    if p.exists():
        print(json.dumps(json.loads(p.read_text(encoding='utf-8')), indent=2, ensure_ascii=False))
    else:
        print('not found')
print('adapter:', 'yes' if (y/'adapter.pt').exists() else 'no')
print('timbre profiles:', 'yes' if (y/'timbre_profiles.pt').exists() else 'no')
print('fidelity refiner:', 'yes' if (y/'fidelity_refiner.pt').exists() else 'no')
print('cache files:', len(list((y/'cache').glob('*.npz'))) if (y/'cache').exists() else 0)
PY
