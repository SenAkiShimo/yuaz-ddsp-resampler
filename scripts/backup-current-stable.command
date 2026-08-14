#!/bin/bash
set -euo pipefail
APP="$HOME/Library/Application Support/YuazDDSP"
OU="$HOME/Library/OpenUtau/Resamplers"
BASE="$HOME/Documents/Yuaz-DDSP-Backups/ai14-preservation"
STAMP="$(date +%Y%m%d-%H%M%S)"
DEST="$BASE/$STAMP"
mkdir -p "$DEST"
echo "created_at=$STAMP" > "$DEST/manifest.txt"
echo "policy=ai.14 non-destructive preservation snapshot" >> "$DEST/manifest.txt"
if [ -d "$APP/0.2.8ai.13" ]; then
  tar -czf "$DEST/ai13-installed-runtime.tar.gz" -C "$APP" "0.2.8ai.13"
  echo "ai13_runtime=yes" >> "$DEST/manifest.txt"
else echo "ai13_runtime=no" >> "$DEST/manifest.txt"; fi
for f in "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.13.sh" "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.13.yaml"; do [ -f "$f" ] && cp -p "$f" "$DEST/" || true; done
SINGERS="$HOME/Library/OpenUtau/Singers"
if [ -d "$SINGERS" ]; then
  mkdir -p "$DEST/voicebank-state"
  python3 - "$SINGERS" "$DEST/voicebank-state" <<'PY'
import re,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]); out=Path(sys.argv[2]); seen=set()
exclude={'cache','cache_ai14','highband_cache','highband_cache_v2','highband_cache_v3','highband_cache_v3_ai14','__pycache__'}
for state in root.rglob('.yuaz-0.2.8ai13'):
    if not state.is_dir(): continue
    bank=state.parent.resolve()
    if bank in seen: continue
    seen.add(bank)
    safe=re.sub(r'[^A-Za-z0-9._-]+','_',bank.name).strip('_') or 'voicebank'
    archive=out/f'{safe}-ai13-state.tar.gz'
    def filt(info): return None if any(x in exclude for x in Path(info.name).parts) else info
    with tarfile.open(archive,'w:gz') as tf: tf.add(state,arcname='.yuaz-0.2.8ai13',filter=filt)
    with (out/'manifest.txt').open('a',encoding='utf-8') as f: f.write(f'{bank}\t{archive.name}\n')
print(f'ai.13 voicebank snapshots: {len(seen)}')
PY
fi
echo "Preservation backup: $DEST"
