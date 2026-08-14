#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${ROOT}/.venv/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$ROOT" <<'PY'
import hashlib,json,sys,tempfile
from pathlib import Path
root=Path(sys.argv[1])
assert (root/'VERSION').read_text().strip()=='0.2.8ai.14'
# Validate a bundled ai.13 source snapshot when the release archive includes one.
manifest=root/'docs/AI13_SNAPSHOT_SHA256.json'
snap=root/'previous_versions/v0.2.8ai.13'
if manifest.is_file() and snap.is_dir():
    m=json.loads(manifest.read_text())
    assert len(m['files'])==m['file_count'] and m['file_count']>=700
    for rel,expected in m['files'].items():
        p=snap/rel; assert p.is_file(), rel
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected, rel
# Active safety policy.
install=(root/'scripts/install-openutau-macos.command').read_text()
assert 'migrate-and-purge-previous.command' not in install
assert 'Yuaz-DDSP-Resampler-v0.2.8ai.13.sh' not in install
assert '47886' in install and 'preserved' in install.lower()
purge=(root/'scripts/purge-previous-version.command').read_text()
assert 'REFUSED' in purge and 'exit 2' in purge
state=(root/'src/yuaz_ddsp_resampler/state.py').read_text()
for name in ('adapter.ai14.pt','timbre_profiles.ai14.pt','fidelity_refiner.ai14.pt','training.ai14.json','deep_validation.ai14.json','highband_profiles_v3.ai14.json'):
    assert name in state, name
resolve=state[state.index('def resolve_active_state'):state.index('def begin_generation')]
assert 'resolve_previous_028ai13_state' not in resolve
assert 'return None' in resolve
prepare=(root/'src/yuaz_ddsp_resampler/prepare.py').read_text(); assert 'cache_ai14' in prepare
# Runtime checkpoint extraction retains only the three resampler components.
import torch
from yuaz_ddsp_resampler.checkpoint_registry import extract_runtime_state
raw={'step':351000,'model':{
 'encoder.a':torch.zeros(2), 'ddsp_decoder.b':torch.zeros(3), 'rvq.c':torch.zeros(4),
 'flow_gen.x':torch.zeros(5), 'mpd.y':torch.zeros(6)}}
out=extract_runtime_state(raw)
assert sorted(out)==['ddsp_decoder.b','encoder.a','rvq.c']
from yuaz_ddsp_resampler.checkpoint_identity import checkpoint_identity
with tempfile.TemporaryDirectory() as td:
    p=Path(td)/'runtime.pt'
    torch.save({'format':'yuaz-ddsp-resampler-runtime-checkpoint-v1','source_checkpoint':'model_351000.pt','source_checkpoint_sha256':'0de7'*16,'source_step':351000,'model':out},p)
    ident=checkpoint_identity(p)
    assert ident['source_step']==351000 and ident['source_checkpoint']=='model_351000.pt'
print('ai.14 release-history policy: OK')
print('ai.14 Deep suffix isolation: OK')
print('ai.14 checkpoint registry helpers: OK')
print('ai.14 no-purge install policy: OK')
PY
python3 -m compileall -q "$ROOT/src"
while IFS= read -r -d '' f; do bash -n "$f"; done < <(find "$ROOT" -type f -name '*.command' -not -path '*/previous_versions/*' -print0)
echo "0.2.8ai.14 checkpoint registry + ai.13 preservation self-test OK"
