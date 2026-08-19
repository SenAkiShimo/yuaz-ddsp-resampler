#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_AI="$ROOT/src/yuaz_ddsp_resampler/ai_vocal_controls.py"
SRC_VOCAL="$ROOT/src/yuaz_ddsp_resampler/vocal_controls.py"
DST_ROOT="$HOME/Library/Application Support/YuazDDSP/0.2.8ai.16"
DST_AI="$DST_ROOT/src/yuaz_ddsp_resampler/ai_vocal_controls.py"
DST_VOCAL="$DST_ROOT/src/yuaz_ddsp_resampler/vocal_controls.py"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="$HOME/Documents/Yuaz-DDSP-Backups/ai16-yv-route-before-$STAMP"
for p in "$SRC_AI" "$SRC_VOCAL" "$DST_AI" "$DST_VOCAL"; do
  [ -f "$p" ] || { echo "Required file missing: $p" >&2; exit 1; }
done
mkdir -p "$BACKUP"
cp "$DST_AI" "$BACKUP/ai_vocal_controls.py"
cp "$DST_VOCAL" "$BACKUP/vocal_controls.py"
cp "$SRC_AI" "$DST_AI"
cp "$SRC_VOCAL" "$DST_VOCAL"
echo "Backup: $BACKUP/ai_vocal_controls.py"
echo "Backup: $BACKUP/vocal_controls.py"
echo "Installed: $DST_AI"
echo "Installed: $DST_VOCAL"
PID="$(lsof -tiTCP:47888 -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$PID" ]; then
  kill "$PID" || true
  echo "Stopped ai16 runtime on port 47888: $PID"
else
  echo "No active ai16 runtime on port 47888."
fi

grep -n "phonation-yv-odd-ap-v2" "$DST_AI"
grep -n "0.58 \* voicing_pos_eff" "$DST_VOCAL"
grep -n "0.90 \* v_pos" "$DST_VOCAL"
