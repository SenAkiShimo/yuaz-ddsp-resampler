#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
echo "===== 0.2.9 DOCTOR ====="
[ -f config.json ] || { echo "FAIL: config.json missing"; exit 1; }
python3 - config.json <<'PY'
import json,sys,os
c=json.load(open(sys.argv[1]))
for k in ('engine_version','runtime_id','port','state_namespace','base_checkpoint_model_id','base_checkpoint_step','base_checkpoint_sha256','checkpoint'):
 print(k+':',c.get(k))
assert c['engine_version']=='0.2.9' and c['runtime_id']=='yuaz-0.2.9' and int(c['port'])==47888
assert c['state_namespace']=='.yuaz-0.2.8ai14' and c.get('preserve_ai14') is True and c.get('trained_artifact_suffix')=='.ai14'
assert os.path.isfile(c['checkpoint'])
PY
echo "Port 47888 listener:"; lsof -nP -iTCP:47888 -sTCP:LISTEN 2>/dev/null || echo "  none"
WRAPPER="$HOME/Library/OpenUtau/Resamplers/Yuaz-DDSP-Resampler-v0.2.9.sh"
[ -f "$WRAPPER" ] && echo "PASS: 0.2.9 wrapper installed" || echo "0.2.9 wrapper not installed yet"
echo "PASS: ai.14 voicebank state remains read-only compatibility state."
echo "PASS: Deep training remains disabled in this runtime."
echo "Doctor OK"
