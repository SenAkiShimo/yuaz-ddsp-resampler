#!/bin/bash
set -euo pipefail
APP="$HOME/Library/Application Support/YuazDDSP"
DEST="$HOME/Library/OpenUtau/Resamplers"
SINGERS="$HOME/Library/OpenUtau/Singers"

# Stop only ai.13 if it is still running.
if [ -x "$APP/0.2.8ai.13/scripts/stop-engine.command" ]; then
  "$APP/0.2.8ai.13/scripts/stop-engine.command" 2>/dev/null || true
fi
pkill -f 'YuazDDSP/0.2.8ai.13.*yuaz_ddsp_resampler.server' 2>/dev/null || true

rm -rf "$APP/0.2.8ai.13"
rm -f "$DEST/Yuaz-DDSP-Resampler-v0.2.8ai.13.sh"
rm -f "$DEST/Yuaz-DDSP-Resampler-v0.2.8ai.13.yaml"

if [ -d "$SINGERS" ]; then
  while IFS= read -r -d '' state; do
    case "$state" in
      *'/.yuaz-0.2.8ai14'|*'/.yuaz-0.2.8ai14/'*)
        echo "REFUSED unsafe path: $state" >&2
        exit 2
        ;;
    esac
    rm -rf "$state"
  done < <(find "$SINGERS" -type d -name '.yuaz-0.2.8ai13' -print0 2>/dev/null)
fi

echo "Removed 0.2.8ai.13 runtime, OpenUtau wrapper/YAML, and .yuaz-0.2.8ai13 voicebank states under OpenUtau/Singers."
echo "PRESERVED: 0.2.8ai.14 runtime, wrapper/YAML, .yuaz-0.2.8ai14 voicebank states, and all ai.14 trained artifacts."
