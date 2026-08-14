#!/bin/bash
set -e
strip_path() { python3 - "$1" <<'PY'
import shlex,sys
p=shlex.split(sys.argv[1].strip()); print(p[0] if p else '')
PY
}
echo "Drop the voicebank root, then press Return:"
read -r RAW
BANK="$(strip_path "$RAW")"
[ -d "$BANK" ] || { echo "Voicebank folder not found: $BANK"; exit 1; }
SAFE="$(python3 - "$BANK" <<'PY'
import sys
from pathlib import Path
text=Path(sys.argv[1]).name
out=''.join(ch if (ch.isalnum() or ch in '-_. ') else '_' for ch in text).strip().replace(' ','_')
print(out or 'voicebank')
PY
)"
BASE="$HOME/Documents/Yuaz-DDSP-Backups/$SAFE"
echo "Backup folder: $BASE"
if [ -d "$BASE" ]; then
  find "$BASE" -maxdepth 1 -mindepth 1 -type d -print | sort -r
else
  echo "No backups yet."
fi
