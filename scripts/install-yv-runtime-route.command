#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_AI="$ROOT/src/yuaz_ddsp_resampler/ai_vocal_controls.py"
SRC_VOCAL="$ROOT/src/yuaz_ddsp_resampler/vocal_controls.py"
SRC_MANIFEST="$ROOT/resampler-manifest.yaml"
DST_ROOT="$HOME/Library/Application Support/YuazDDSP/0.2.8ai.16"
DST_AI="$DST_ROOT/src/yuaz_ddsp_resampler/ai_vocal_controls.py"
DST_VOCAL="$DST_ROOT/src/yuaz_ddsp_resampler/vocal_controls.py"
WRAPPER="$HOME/Library/OpenUtau/Resamplers/Yuaz-DDSP-Resampler-v0.2.8ai.16.sh"
DST_MANIFEST="${WRAPPER%.sh}.yaml"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="$HOME/Documents/Yuaz-DDSP-Backups/ai16-yv-route-before-$STAMP"
for p in "$SRC_AI" "$SRC_VOCAL" "$SRC_MANIFEST" "$DST_AI" "$DST_VOCAL" "$WRAPPER"; do
  [ -f "$p" ] || { echo "Required file missing: $p" >&2; exit 1; }
done
mkdir -p "$BACKUP"
cp "$DST_AI" "$BACKUP/ai_vocal_controls.py"
cp "$DST_VOCAL" "$BACKUP/vocal_controls.py"
if [ -f "$DST_MANIFEST" ]; then
  cp "$DST_MANIFEST" "$BACKUP/resampler-manifest.yaml"
fi
cp "$SRC_AI" "$DST_AI"
cp "$SRC_VOCAL" "$DST_VOCAL"
cp "$SRC_MANIFEST" "$DST_MANIFEST"
echo "Backup: $BACKUP/ai_vocal_controls.py"
echo "Backup: $BACKUP/vocal_controls.py"
if [ -f "$BACKUP/resampler-manifest.yaml" ]; then
  echo "Backup: $BACKUP/resampler-manifest.yaml"
fi
echo "Installed: $DST_AI"
echo "Installed: $DST_VOCAL"
echo "Installed: $DST_MANIFEST"
PID="$(lsof -tiTCP:47888 -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$PID" ]; then
  kill "$PID" || true
  echo "Stopped ai16 runtime on port 47888: $PID"
else
  echo "No active ai16 runtime on port 47888."
fi

grep -n "phonation-yv-odd-ap-v2" "$DST_AI"
grep -n 'voicing_pos_scale = carrier("voicing", 1.00)' "$DST_VOCAL"
grep -n "0.72 \* voicing_pos_eff" "$DST_VOCAL"
grep -n "0.52 \* v_pos_ap" "$DST_VOCAL"
grep -n "0.92 \* v_pos" "$DST_VOCAL"
awk '
  /^  yvc:/ {in_yv=1}
  in_yv {print NR ":" $0}
  in_yv && /^    flag: YV$/ {found=1; exit}
  END {if (!found) exit 1}
' "$DST_MANIFEST"
echo "YV manifest dispatch verified: yvc -> flag YV"
echo "YV positive carrier verified: scale 1.00 with stronger body, periodicity, and closure."
echo "Restart OpenUtau before listening tests."
