#!/bin/bash
set -e
strip_path() { python3 - "$1" <<'PY'
import shlex,sys
p=shlex.split(sys.argv[1].strip()); print(p[0] if p else '')
PY
}
echo "This removes ONLY alpha.8 RC1/RC2 OpenUtau entries, Downloads folders, and the selected voicebank's RC1/RC2 state."
echo "RC3.1 and RC3.2 are preserved."
echo "Drop the voicebank root, then press Return:"
read -r RAW
BANK="$(strip_path "$RAW")"
[ -d "$BANK" ] || { echo "Voicebank folder not found: $BANK"; exit 1; }
pkill -f 'yuaz-ddsp-resampler-v0.2.7-alpha.8-rc.1' 2>/dev/null || true
pkill -f 'yuaz-ddsp-resampler-v0.2.7-alpha.8-rc.2' 2>/dev/null || true
rm -f \
  "$HOME/Library/OpenUtau/Resamplers/Yuaz-DDSP-Resampler-v0.2.7-alpha.8-rc.1.sh" \
  "$HOME/Library/OpenUtau/Resamplers/Yuaz-DDSP-Resampler-v0.2.7-alpha.8-rc.1.yaml" \
  "$HOME/Library/OpenUtau/Resamplers/Yuaz-DDSP-Resampler-v0.2.7-alpha.8-rc.2.sh" \
  "$HOME/Library/OpenUtau/Resamplers/Yuaz-DDSP-Resampler-v0.2.7-alpha.8-rc.2.yaml"
rm -rf \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.7-alpha.8-rc.1" \
  "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.7-alpha.8-rc.2" \
  "$BANK/.yuaz-alpha8-rc1" \
  "$BANK/.yuaz-alpha8-rc2"
echo "RC1/RC2 cleanup complete. RC3.1/RC3.2 were not touched."
