#!/bin/bash
set -euo pipefail
APP="$HOME/Library/Application Support/YuazDDSP"
OU="$HOME/Library/OpenUtau/Resamplers"
BASE="$HOME/Documents/Yuaz-DDSP-Backups/engine-snapshots"
STAMP="$(date +%Y%m%d-%H%M%S)"
DEST="$BASE/$STAMP-before-0.2.8ai.12"
mkdir -p "$DEST"

snapshot_runtime() {
  local label="$1" runtime="$2" wrap="$3" yaml="$4"
  echo "[$label] runtime=$runtime" >> "$DEST/manifest.txt"
  echo "[$label] wrapper=$wrap" >> "$DEST/manifest.txt"
  if [ -d "$runtime" ]; then
    tar -czf "$DEST/${label}-installed-runtime.tar.gz" -C "$APP" "$(basename "$runtime")"
    echo "[$label] runtime_snapshot=yes" >> "$DEST/manifest.txt"
  else
    echo "[$label] runtime_snapshot=no" >> "$DEST/manifest.txt"
  fi
  [ -f "$wrap" ] && cp -p "$wrap" "$DEST/" || true
  [ -f "$yaml" ] && cp -p "$yaml" "$DEST/" || true
}

{
  echo "created_at=$STAMP"
  echo "policy=pre-purge snapshots; 0.2.8ai.11, 0.2.8ai.10, 0.2.8ai.9, 0.2.8ai.8, 0.2.8ai.7, 0.2.8ai.6, 0.2.8ai.5, 0.2.8ai.4, 0.2.8ai.3, 0.2.8ai.2, 0.2.8ai.1, 0.2.8ai, 0.2.7 AI.3 and RC4.2 are not stopped, removed, overwritten, or modified"
} > "$DEST/manifest.txt"

snapshot_runtime \
  "rc4.2" \
  "$APP/0.2.7-alpha.8-rc.4.2" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.7-alpha.8-rc.4.2.sh" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.7-alpha.8-rc.4.2.yaml"


snapshot_runtime \
  "0.2.8ai.11" \
  "$APP/0.2.8ai.11" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.11.sh" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.11.yaml"

snapshot_runtime \
  "0.2.8ai.10" \
  "$APP/0.2.8ai.10" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.10.sh" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.10.yaml"

snapshot_runtime \
  "0.2.8ai.9" \
  "$APP/0.2.8ai.9" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.9.sh" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.9.yaml"

snapshot_runtime \
  "0.2.8ai.8" \
  "$APP/0.2.8ai.8" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.8.sh" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.8.yaml"

snapshot_runtime \
  "0.2.8ai.7" \
  "$APP/0.2.8ai.7" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.7.sh" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.7.yaml"

snapshot_runtime \
  "0.2.8ai.6" \
  "$APP/0.2.8ai.6" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.6.sh" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.6.yaml"

snapshot_runtime \
  "0.2.8ai.5" \
  "$APP/0.2.8ai.5" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.5.sh" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.5.yaml"

snapshot_runtime \
  "0.2.8ai.4" \
  "$APP/0.2.8ai.4" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.4.sh" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.4.yaml"

snapshot_runtime \
  "0.2.8ai.3" \
  "$APP/0.2.8ai.3" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.3.sh" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.3.yaml"

snapshot_runtime \
  "0.2.8ai.2" \
  "$APP/0.2.8ai.2" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.2.sh" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.2.yaml"

snapshot_runtime \
  "0.2.8ai.1" \
  "$APP/0.2.8ai.1" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.1.sh" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.1.yaml"

snapshot_runtime \
  "0.2.8ai" \
  "$APP/0.2.8ai" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.sh" \
  "$OU/Yuaz-DDSP-Resampler-v0.2.8ai.yaml"

snapshot_runtime \
  "0.2.7-ai3" \
  "$APP/0.2.7-alpha.8-rc.4.3-ai.3" \
  "$OU/Yuaz-DDSP-Resampler-AI-v0.2.7-alpha.8-rc.4.3-ai.3.sh" \
  "$OU/Yuaz-DDSP-Resampler-AI-v0.2.7-alpha.8-rc.4.3-ai.3.yaml"

# Snapshot every current ai.11 voicebank state before the installer is allowed to purge it.
# Derived caches are excluded so this remains a practical safety copy even for large banks.
SINGERS="$HOME/Library/OpenUtau/Singers"
if [ -d "$SINGERS" ]; then
  mkdir -p "$DEST/voicebank-state"
  python3 - "$SINGERS" "$DEST/voicebank-state" <<'PY_STATE'
import os, re, sys, tarfile
from pathlib import Path
root=Path(sys.argv[1]).expanduser()
out=Path(sys.argv[2]).expanduser()
exclude={"cache","highband_cache","highband_cache_v2","highband_cache_v3","__pycache__"}
seen=set()
for state in root.rglob('.yuaz-0.2.8ai11'):
    if not state.is_dir():
        continue
    bank=state.parent.resolve()
    if bank in seen:
        continue
    seen.add(bank)
    safe=re.sub(r'[^A-Za-z0-9._-]+','_',bank.name).strip('_') or 'voicebank'
    archive=out/f'{safe}-0.2.8ai11-state.tar.gz'
    def filt(info):
        rel=Path(info.name)
        if any(part in exclude for part in rel.parts):
            return None
        return info
    with tarfile.open(archive,'w:gz') as tf:
        tf.add(state, arcname='.yuaz-0.2.8ai11', filter=filt)
    with (out/'manifest.txt').open('a',encoding='utf-8') as f:
        f.write(f'{bank}\t{archive.name}\n')
print(f'Voicebank ai.11 safety snapshots: {len(seen)}')
PY_STATE
fi

for CAND in \
  "$HOME/Downloads/Yuaz_DDSP_v0.2.8ai.11_BRANCH_REUPLOAD.zip" \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.8ai.11.zip" \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.8ai.10-highband-hotfix2.zip" \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.8ai.10-highband-hotfix1.zip" \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.8ai.10.zip" \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.8ai.9-fixed.zip" \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.8ai.9.zip" \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.8ai.8.zip" \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.8ai.7.zip" \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.8ai.6.zip" \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.8ai.5.zip" \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.8ai.4.zip" \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.8ai.3.zip" \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.8ai.2.zip" \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.8ai.1.zip" \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.8ai.zip" \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.7-alpha.8-rc.4.2.zip" \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.7-alpha.8-rc.4.3-ai.3.zip" \
  "$HOME/Documents/Yuaz-DDSP-Backups/control-models/ai_control_foundation-v2-Chinese-Core.pt" \
  "$HOME/Documents/Yuaz-DDSP-Backups/control-models/ai_gender_foundation-v1-VocalSet.pt" \
  "$HOME/Documents/Yuaz-DDSP-Backups/control-models/ai_phonation_foundation-v1-PhonationModes-MOCHA.pt" \
  "$HOME/Documents/Yuaz-DDSP-Backups/control-models/ai_mouth_foundation-v1-MOCHA.pt"; do
  [ -f "$CAND" ] && cp -p "$CAND" "$DEST/" || true
done

( cd "$DEST" && find . -maxdepth 1 -type f ! -name SHA256SUMS.txt -print0 | xargs -0 shasum -a 256 > SHA256SUMS.txt ) || true
echo "$DEST" > "$BASE/LATEST_PRE_0.2.8AI12_ENGINE_SNAPSHOT.txt"
echo "Pre-0.2.8ai.12 engine snapshot created: $DEST"
echo "Snapshot includes current 0.2.8ai.11 and available predecessors before migration/purge. Previous installed versions may be removed after successful state migration. Included when present: 0.2.8ai.11 + 0.2.8ai.10 + 0.2.8ai.9 + 0.2.8ai.8 + 0.2.8ai.7 + 0.2.8ai.6 + 0.2.8ai.5 + 0.2.8ai.4 + 0.2.8ai.3 + 0.2.8ai.2 + 0.2.8ai.1 + 0.2.8ai + 0.2.7 AI.3 + RC4.2 runtimes/wrappers + existing voicebank state."
