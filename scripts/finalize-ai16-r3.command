#!/bin/bash
set -euo pipefail
RESAMPLERS="$HOME/Library/OpenUtau/Resamplers"
KEEP="$RESAMPLERS/Yuaz-DDSP-Resampler-v0.2.8ai.16-r3.sh"
KEEP_MANIFEST="${KEEP%.sh}.yaml"

[ -f "$KEEP" ] || { echo "Current r3 wrapper missing: $KEEP" >&2; exit 1; }
[ -f "$KEEP_MANIFEST" ] || { echo "Current r3 manifest missing: $KEEP_MANIFEST" >&2; exit 1; }

for stem in \
  "Yuaz-DDSP-Resampler-v0.2.8ai.16" \
  "Yuaz-DDSP-Resampler-v0.2.8ai.16-r1" \
  "Yuaz-DDSP-Resampler-v0.2.8ai.16-r2"
do
  rm -f "$RESAMPLERS/$stem.sh" "$RESAMPLERS/$stem.yaml"
done

chmod +x "$KEEP"

echo "Kept current version:"
echo "  $KEEP"
echo "  $KEEP_MANIFEST"
echo "Removed old OpenUtau wrappers: ai16, r1, r2 (when present)."
echo "Shared runtime preserved: $HOME/Library/Application Support/YuazDDSP/0.2.8ai.16"
echo "Voicebank state and OpenUtau audio caches were not touched."
