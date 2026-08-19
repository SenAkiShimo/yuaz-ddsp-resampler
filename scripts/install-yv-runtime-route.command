#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src/yuaz_ddsp_resampler/ai_vocal_controls.py"
DST_ROOT="$HOME/Library/Application Support/YuazDDSP/0.2.8ai.16"
DST="$DST_ROOT/src/yuaz_ddsp_resampler/ai_vocal_controls.py"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="$HOME/Documents/Yuaz-DDSP-Backups/ai16-yv-route-before-$STAMP"
[ -f "$SRC" ] || { echo "Source file missing: $SRC" >&2; exit 1; }
[ -f "$DST" ] || { echo "Installed ai16 file missing: $DST" >&2; exit 1; }
mkdir -p "$BACKUP"
cp "$DST" "$BACKUP/ai_vocal_controls.py"
cp "$SRC" "$DST"
echo "Backup: $BACKUP/ai_vocal_controls.py"
echo "Installed: $DST"
PID="$(lsof -tiTCP:47888 -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$PID" ]; then
  kill "$PID" || true
  echo "Stopped ai16 runtime on port 47888: $PID"
else
  echo "No active ai16 runtime on port 47888."
fi

grep -n "phonation-yv-odd-ap-v1" "$DST"
