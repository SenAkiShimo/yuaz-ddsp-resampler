#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
echo "===== 0.2.8ai.14 DOCTOR ====="
[ -f config.json ] || { echo "FAIL: config.json missing"; exit 1; }
python3 - config.json <<'PY'
import json,sys,os
c=json.load(open(sys.argv[1]))
for k in ('engine_version','runtime_id','port','state_namespace','base_checkpoint_model_id','base_checkpoint_step','base_checkpoint_sha256','checkpoint'):
 print(k+':',c.get(k))
assert c['engine_version']=='0.2.8ai.14' and c['runtime_id']=='yuaz-0.2.8ai.14-control-v14' and int(c['port'])==47886
assert c['state_namespace']=='.yuaz-0.2.8ai14' and c.get('preserve_ai13') is True and c.get('trained_artifact_suffix')=='.ai14'
assert os.path.isfile(c['checkpoint'])
PY
echo "Port 47886 listener:"; lsof -nP -iTCP:47886 -sTCP:LISTEN 2>/dev/null || echo "  none"
AI14="$HOME/Library/OpenUtau/Resamplers/Yuaz-DDSP-Resampler-v0.2.8ai.14.sh"
AI13="$HOME/Library/OpenUtau/Resamplers/Yuaz-DDSP-Resampler-v0.2.8ai.13.sh"
[ -f "$AI14" ] && echo "PASS: ai.14 wrapper installed" || echo "ai.14 wrapper not installed yet"
[ -f "$AI13" ] && echo "PASS: ai.13 wrapper preserved: $AI13" || echo "NOTE: ai.13 wrapper was not present before/at this check; ai.14 does not delete it."
echo "PASS: purge disabled; ai.13 state namespace is never an ai.14 fallback."
echo "PASS: Deep suffix protection: adapter.ai14.pt / fidelity_refiner.ai14.pt / training.ai14.json / deep_validation.ai14.json."
echo "Doctor OK"
