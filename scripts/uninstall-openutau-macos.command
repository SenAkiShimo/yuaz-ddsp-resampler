#!/bin/bash
set -euo pipefail
DEST="$HOME/Library/OpenUtau/Resamplers"
APP="$HOME/Library/Application Support/YuazDDSP"
VER="0.2.8ai.14"
RUNTIME="$APP/$VER"
[ -x "$RUNTIME/scripts/stop-engine.command" ] && "$RUNTIME/scripts/stop-engine.command" 2>/dev/null || true
rm -f "$DEST/Yuaz-DDSP-Resampler-v$VER.sh" "$DEST/Yuaz-DDSP-Resampler-v$VER.yaml"
rm -rf "$RUNTIME"
echo "Removed only 0.2.8ai.14 runtime/wrapper."
echo "0.2.8ai.5, 0.2.8ai.4, 0.2.8ai.3, 0.2.8ai.2, 0.2.8ai.1, 0.2.8ai, 0.2.7 AI.3, RC4.2, and every voicebank generation outside .yuaz-0.2.8ai14 were preserved."
