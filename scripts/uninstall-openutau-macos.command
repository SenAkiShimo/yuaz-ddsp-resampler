#!/bin/bash
set -e
DEST="$HOME/Library/OpenUtau/Resamplers"
rm -f "$DEST/Yuaz-DDSP-Resampler-v0.2.7-alpha.8-rc.3.2.sh" "$DEST/Yuaz-DDSP-Resampler-v0.2.7-alpha.8-rc.3.2.yaml"
echo "Removed only alpha.8 RC3.2 OpenUtau entries."
echo "Other resamplers and voicebank adaptation data were preserved."
