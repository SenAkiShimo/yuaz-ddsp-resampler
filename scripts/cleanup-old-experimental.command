#!/bin/bash
set -e
DEST="$HOME/Library/OpenUtau/Resamplers"
for V in 2 3 4; do rm -f "$DEST/Yuaz-DDSP-Resampler-v0.2.7-alpha.$V.sh" "$DEST/Yuaz-DDSP-Resampler-v0.2.7-alpha.$V.yaml"; done
pkill -f 'yuaz_ddsp_resampler.server.*alpha.2' 2>/dev/null || true
pkill -f 'yuaz_ddsp_resampler.server.*alpha.3' 2>/dev/null || true
pkill -f 'yuaz_ddsp_resampler.server.*alpha.4' 2>/dev/null || true
rm -rf "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.7-alpha.2" "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.7-alpha.3" "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.7-alpha.4"
echo "Removed alpha.2, alpha.3, and alpha.4 OpenUtau entries and Downloads folders only."
echo "alpha.1, alpha.5, alpha.6, alpha.8 RC3.1, alpha.8 RC3.2, and every voicebank adaptation folder were preserved."
