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
echo "Drop the voicebank root whose Yuaz adaptation should be removed, then press Return:"
read -r RAW
BANK="$(strip_path "$RAW")"
if [ ! -d "$BANK" ]; then
  echo "Voicebank folder not found: $BANK"
  exit 1
fi
python3 - "$ROOT" "$BANK" <<'PY'
import json, shutil, sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
bank=Path(sys.argv[2]).resolve()
registry=root/'voicebank_registry.json'
if registry.exists():
    try:
        data=json.loads(registry.read_text(encoding='utf-8'))
    except Exception:
        data={'format':1,'samples':{}}
    samples=data.get('samples',{})
    data['samples']={k:v for k,v in samples.items() if Path(v.get('voicebank_root','')).expanduser().resolve()!=bank}
    registry.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
y=bank/'.yuaz'
if y.exists():
    shutil.rmtree(y)
print(f'Removed adaptation data for: {bank}')
PY
