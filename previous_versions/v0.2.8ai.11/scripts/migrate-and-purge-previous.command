#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "Run setup-macos.command first."; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
SINGERS="$HOME/Library/OpenUtau/Singers"
APP="$HOME/Library/Application Support/YuazDDSP"
DEST="$HOME/Library/OpenUtau/Resamplers"
REG="$APP/state/voicebank_registry-0.2.8ai11.json"

echo "Yuaz 0.2.8ai.11 migration + previous-version purge"
echo "First migrates the newest valid prepared state into .yuaz-0.2.8ai11."
echo "0.2.8ai.10 is the preferred migration source; adapter/Fidelity/articulation/high-band weights are preserved."
echo "Derived analysis/high-band caches are intentionally not copied and will rebuild on first use."
echo "Only after all migrations validate does it remove older Yuaz runtimes/wrappers/state containers."
echo "Source WAV/OTO, datasets, and ~/Documents/Yuaz-DDSP-Backups are never deleted."

"$PY" - "$SINGERS" "$REG" <<'PY'
import json, sys
from pathlib import Path
from yuaz_ddsp_resampler.state import (
    STATE_CONTAINER, PREVIOUS_028AI10_STATE_CONTAINER, PREVIOUS_028AI9_STATE_CONTAINER,
    PREVIOUS_028AI8_STATE_CONTAINER, PREVIOUS_028AI7_STATE_CONTAINER, PREVIOUS_028AI6_STATE_CONTAINER,
    PREVIOUS_028AI5_STATE_CONTAINER, PREVIOUS_028AI4_STATE_CONTAINER, PREVIOUS_028AI3_STATE_CONTAINER,
    PREVIOUS_028AI2_STATE_CONTAINER, PREVIOUS_028AI1_STATE_CONTAINER, PREVIOUS_028_STATE_CONTAINER,
    PREDECESSOR_AI_STATE_CONTAINER, STABLE_STATE_CONTAINER, LEGACY_STATE,
    resolve_active_state, resolve_ai_state, begin_generation, generation_dir, clone_state,
    commit_generation, write_local_registry, merge_global_registry, validate_state,
)
root=Path(sys.argv[1]).expanduser()
global_registry=Path(sys.argv[2]).expanduser()
legacy_names=(
    PREVIOUS_028AI10_STATE_CONTAINER, PREVIOUS_028AI9_STATE_CONTAINER, PREVIOUS_028AI8_STATE_CONTAINER,
    PREVIOUS_028AI7_STATE_CONTAINER, PREVIOUS_028AI6_STATE_CONTAINER, PREVIOUS_028AI5_STATE_CONTAINER,
    PREVIOUS_028AI4_STATE_CONTAINER, PREVIOUS_028AI3_STATE_CONTAINER, PREVIOUS_028AI2_STATE_CONTAINER,
    PREVIOUS_028AI1_STATE_CONTAINER, PREVIOUS_028_STATE_CONTAINER, PREDECESSOR_AI_STATE_CONTAINER,
    STABLE_STATE_CONTAINER, LEGACY_STATE,
)
if not root.is_dir():
    print('No OpenUtau Singers directory; nothing to migrate.')
    raise SystemExit(0)
banks=[]
for p in root.rglob('*'):
    if p.is_dir() and p.name in legacy_names:
        bank=p.parent.resolve()
        if bank not in banks:
            banks.append(bank)
print(f'Detected voicebanks with legacy Yuaz state: {len(banks)}')

def rewrite_state_refs(staging, source, final, bank):
    src_abs=str(source.resolve()); dst_abs=str(final.resolve())
    try: src_rel=source.resolve().relative_to(bank.resolve()).as_posix()
    except Exception: src_rel=''
    dst_rel=final.resolve().relative_to(bank.resolve()).as_posix()
    def rw(v):
        if isinstance(v,str):
            z=v.replace(src_abs,dst_abs)
            if src_rel: z=z.replace(src_rel,dst_rel)
            return z
        if isinstance(v,list): return [rw(x) for x in v]
        if isinstance(v,dict): return {k:rw(x) for k,x in v.items()}
        return v
    for jp in staging.rglob('*.json'):
        try: data=json.loads(jp.read_text(encoding='utf-8'))
        except Exception: continue
        jp.write_text(json.dumps(rw(data),indent=2,ensure_ascii=False),encoding='utf-8')

fail=[]
for bank in banks:
    current,_=resolve_ai_state(bank,verify=True)
    if current is not None:
        print('KEEP current:',bank,'->',current)
        continue
    source,info=resolve_active_state(bank,allow_legacy=True,verify=True)
    if source is None:
        fail.append((str(bank),'no valid source state'))
        continue
    try:
        generation,staging=begin_generation(bank,'migrated')
        final=generation_dir(bank,generation)
        clone_state(source,staging,link_caches=False,skip_caches=True)
        rewrite_state_refs(staging,source,final,bank)
        final,payload=commit_generation(
            bank,generation,staging,
            reason=f'migrated-from-{info.get("source")}',
            acoustic_base='0.2.8ai.11-dual-rate-48k-ddsp-body',
        )
        validate_state(final,verify_hashes=True)
        reg=write_local_registry(bank,final)
        merge_global_registry(global_registry,reg)
        print('MIGRATED:',bank,'<-',info.get('source'),'->',final)
    except Exception as exc:
        fail.append((str(bank),repr(exc)))
if fail:
    print('\nMigration failed; NO previous state containers should be purged.')
    for bank,err in fail: print('FAIL',bank,err)
    raise SystemExit(2)
print('All detected voicebank migrations validated.')
PY

# Stop old engines before deleting their runtimes.
for D in \
  "$APP/0.2.8ai.10" "$APP/0.2.8ai.9" "$APP/0.2.8ai.8" "$APP/0.2.8ai.7" "$APP/0.2.8ai.6" "$APP/0.2.8ai.5" "$APP/0.2.8ai.4" "$APP/0.2.8ai.3" \
  "$APP/0.2.8ai.2" "$APP/0.2.8ai.1" "$APP/0.2.8ai" \
  "$APP/0.2.7-alpha.8-rc.4.3-ai.3" "$APP/0.2.7-alpha.8-rc.4.2"; do
  [ -e "$D" ] || continue
  [ -x "$D/scripts/stop-engine.command" ] && "$D/scripts/stop-engine.command" 2>/dev/null || true
done
rm -rf \
  "$APP/0.2.8ai.10" "$APP/0.2.8ai.9" "$APP/0.2.8ai.8" "$APP/0.2.8ai.7" "$APP/0.2.8ai.6" "$APP/0.2.8ai.5" "$APP/0.2.8ai.4" "$APP/0.2.8ai.3" \
  "$APP/0.2.8ai.2" "$APP/0.2.8ai.1" "$APP/0.2.8ai" \
  "$APP/0.2.7-alpha.8-rc.4.3-ai.3" "$APP/0.2.7-alpha.8-rc.4.2"
rm -rf "$APP"/.0.2.8ai*-installing-* 2>/dev/null || true

if [ -d "$DEST" ]; then
  find "$DEST" -maxdepth 1 -type f \
    \( -name 'Yuaz-DDSP-Resampler*.sh' -o -name 'Yuaz-DDSP-Resampler*.yaml' \) \
    ! -name 'Yuaz-DDSP-Resampler-v0.2.8ai.11.sh' \
    ! -name 'Yuaz-DDSP-Resampler-v0.2.8ai.11.yaml' -delete
fi

rm -f "$APP/state"/voicebank_registry-0.2.8ai10.json \
      "$APP/state"/voicebank_registry-0.2.8ai9.json \
      "$APP/state"/voicebank_registry-0.2.8ai8.json \
      "$APP/state"/voicebank_registry-0.2.8ai7.json \
      "$APP/state"/voicebank_registry-0.2.8ai6.json \
      "$APP/state"/voicebank_registry-0.2.8ai5.json \
      "$APP/state"/voicebank_registry-0.2.8ai4.json \
      "$APP/state"/voicebank_registry-0.2.8ai3.json \
      "$APP/state"/voicebank_registry-0.2.8ai2.json \
      "$APP/state"/voicebank_registry-0.2.8ai1.json \
      "$APP/state"/voicebank_registry-0.2.8ai.json 2>/dev/null || true

if [ -d "$SINGERS" ]; then
  find "$SINGERS" -type d \( \
    -name '.yuaz-0.2.8ai10' -o -name '.yuaz-0.2.8ai9' -o -name '.yuaz-0.2.8ai8' -o -name '.yuaz-0.2.8ai7' -o -name '.yuaz-0.2.8ai6' -o -name '.yuaz-0.2.8ai5' -o -name '.yuaz-0.2.8ai4' -o \
    -name '.yuaz-0.2.8ai3' -o -name '.yuaz-0.2.8ai2' -o -name '.yuaz-0.2.8ai1' -o \
    -name '.yuaz-0.2.8ai' -o -name '.yuaz-alpha8-rc4-3-ai3' -o \
    -name '.yuaz-alpha8-rc3-3' -o -name '.yuaz-alpha8-rc3-2' \) -prune -exec rm -rf {} +
fi

echo "Previous installed Yuaz versions and migrated legacy state containers removed."
echo "Preserved: .yuaz-0.2.8ai11, source WAV/OTO, datasets, and ~/Documents/Yuaz-DDSP-Backups."
