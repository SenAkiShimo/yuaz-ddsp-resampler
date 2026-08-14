#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/control_models"
mkdir -p "$DEST"
import_one() {
  local target="$1"; shift
  [ -f "$DEST/$target" ] && return 0
  local c
  for c in "$@"; do
    if [ -f "$c" ]; then
      cp -p "$c" "$DEST/$target"
      echo "Imported existing learned pack: $target <- $c"
      return 0
    fi
  done
  return 0
}
B="$HOME/Documents/Yuaz-DDSP-Backups/control-models"
for V in 0.2.8ai.5 0.2.8ai.4 0.2.8ai.3 0.2.8ai.2 0.2.8ai.1 0.2.8ai; do
  D="$HOME/Downloads/yuaz-ddsp-resampler-v$V/control_models"
  A="$HOME/Library/Application Support/YuazDDSP/$V/control_models"
  case "$V" in
    0.2.8ai.5) D5="$D"; A5="$A";;
    0.2.8ai.4) D4="$D"; A4="$A";;
    0.2.8ai.3) D3="$D"; A3="$A";;
    0.2.8ai.2) D2="$D"; A2="$A";;
    0.2.8ai.1) D1="$D"; A1="$A";;
    0.2.8ai) D0="$D"; A0="$A";;
  esac
done
import_one ai_control_foundation-v2.pt \
  "$D5/ai_control_foundation-v2.pt" "$A5/ai_control_foundation-v2.pt" \
  "$B/0.2.8ai.5/ai_control_foundation-v2.pt" \
  "$D4/ai_control_foundation-v2.pt" "$A4/ai_control_foundation-v2.pt" \
  "$D3/ai_control_foundation-v2.pt" "$A3/ai_control_foundation-v2.pt" \
  "$D2/ai_control_foundation-v2.pt" "$A2/ai_control_foundation-v2.pt" \
  "$D1/ai_control_foundation-v2.pt" "$A1/ai_control_foundation-v2.pt" \
  "$B/ai_control_foundation-v2-Chinese-Core.pt"
import_one ai_gender_foundation-v1.pt \
  "$D5/ai_gender_foundation-v1.pt" "$A5/ai_gender_foundation-v1.pt" \
  "$B/0.2.8ai.5/ai_gender_foundation-v1.pt" \
  "$D4/ai_gender_foundation-v1.pt" "$A4/ai_gender_foundation-v1.pt" \
  "$D3/ai_gender_foundation-v1.pt" "$A3/ai_gender_foundation-v1.pt" \
  "$D2/ai_gender_foundation-v1.pt" "$A2/ai_gender_foundation-v1.pt" \
  "$D1/ai_gender_foundation-v1.pt" "$A1/ai_gender_foundation-v1.pt" \
  "$B/ai_gender_foundation-v1-VocalSet.pt"
import_one ai_phonation_foundation-v1.pt \
  "$D5/ai_phonation_foundation-v1.pt" "$A5/ai_phonation_foundation-v1.pt" \
  "$B/0.2.8ai.5/ai_phonation_foundation-v1.pt" \
  "$D4/ai_phonation_foundation-v1.pt" "$A4/ai_phonation_foundation-v1.pt" \
  "$D3/ai_phonation_foundation-v1.pt" "$A3/ai_phonation_foundation-v1.pt" \
  "$B/ai_phonation_foundation-v1-PhonationModes-MOCHA.pt" \
  "$B/ai_phonation_foundation-v1-VQS-MOCHA.pt"
import_one ai_mouth_foundation-v1.pt \
  "$D5/ai_mouth_foundation-v1.pt" "$A5/ai_mouth_foundation-v1.pt" \
  "$B/0.2.8ai.5/ai_mouth_foundation-v1.pt" \
  "$D4/ai_mouth_foundation-v1.pt" "$A4/ai_mouth_foundation-v1.pt" \
  "$D3/ai_mouth_foundation-v1.pt" "$A3/ai_mouth_foundation-v1.pt" \
  "$B/ai_mouth_foundation-v1-MOCHA.pt"
