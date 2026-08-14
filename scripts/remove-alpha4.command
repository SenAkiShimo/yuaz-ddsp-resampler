#!/bin/bash
set -e
DEST="$HOME/Library/OpenUtau/Resamplers"
rm -f "$DEST/Yuaz-DDSP-Resampler-v0.2.7-alpha.4.sh" "$DEST/Yuaz-DDSP-Resampler-v0.2.7-alpha.4.yaml"
pkill -f 'yuaz_ddsp_resampler.server.*alpha.4' 2>/dev/null || true
rm -rf "$HOME/Downloads/yuaz-ddsp-resampler-v0.2.7-alpha.4"
echo "Removed alpha.4 OpenUtau entries and ~/Downloads alpha.4 folder only."
echo "alpha.1, alpha.3, alpha.5, alpha.6, and every voicebank .yuaz folder were left untouched."
