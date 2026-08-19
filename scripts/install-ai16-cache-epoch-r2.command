#!/bin/bash
set -euo pipefail
RESAMPLERS="$HOME/Library/OpenUtau/Resamplers"
SRC="$RESAMPLERS/Yuaz-DDSP-Resampler-v0.2.8ai.16.sh"
SRC_MANIFEST="${SRC%.sh}.yaml"
DST="$RESAMPLERS/Yuaz-DDSP-Resampler-v0.2.8ai.16-r2.sh"
DST_MANIFEST="${DST%.sh}.yaml"
BACKUP_ROOT="$HOME/Documents/Yuaz-DDSP-Backups"
STAMP="$(date +%Y%m%d-%H%M%S)"

[ -f "$SRC" ] || { echo "Installed ai16 wrapper missing: $SRC" >&2; exit 1; }
[ -f "$SRC_MANIFEST" ] || { echo "Installed ai16 manifest missing: $SRC_MANIFEST" >&2; exit 1; }
mkdir -p "$RESAMPLERS" "$BACKUP_ROOT"

if [ -e "$DST" ] || [ -e "$DST_MANIFEST" ]; then
  BACKUP="$BACKUP_ROOT/ai16-r2-wrapper-before-$STAMP"
  mkdir -p "$BACKUP"
  [ -f "$DST" ] && cp "$DST" "$BACKUP/$(basename "$DST")"
  [ -f "$DST_MANIFEST" ] && cp "$DST_MANIFEST" "$BACKUP/$(basename "$DST_MANIFEST")"
  echo "Existing r2 wrapper backed up to: $BACKUP"
fi

cp "$SRC" "$DST"
cp "$SRC_MANIFEST" "$DST_MANIFEST"
chmod +x "$DST"

printf '%s\n' \
  "Installed new OpenUtau cache identity only:" \
  "  $DST" \
  "  $DST_MANIFEST" \
  "" \
  "The original ai16 wrapper remains untouched:" \
  "  $SRC" \
  "" \
  "Both wrappers call the same installed ai16 runtime." \
  "The -r2 filename exists only so OpenUtau creates fresh resampler cache keys after runtime-code changes." \
  "No voicebank state, Deep data, model pack, or old OpenUtau cache was deleted." \
  "" \
  "Restart OpenUtau, select Yuaz-DDSP-Resampler-v0.2.8ai.16-r2 as the resampler, then test YV0 / YV100 / YV-100."
