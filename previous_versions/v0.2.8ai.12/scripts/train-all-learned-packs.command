#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/scripts/import-existing-control-packs.command"
echo "Yuaz 0.2.8ai.12 — Train ALL currently available learned-control packs"
echo "Existing GTSinger YB/YF/YX/YP foundation is reused and is NOT retrained."
echo
if [ ! -f "$ROOT/control_models/ai_gender_foundation-v1.pt" ]; then "$ROOT/scripts/train-ai-gender-foundation.command"; else echo "Reuse existing Gender pack: control_models/ai_gender_foundation-v1.pt"; fi
if [ ! -f "$ROOT/control_models/ai_phonation_foundation-v1.pt" ]; then "$ROOT/scripts/train-ai-phonation-foundation.command"; else echo "Reuse existing Phonation pack: control_models/ai_phonation_foundation-v1.pt"; fi
if [ ! -f "$ROOT/control_models/ai_mouth_foundation-v1.pt" ]; then "$ROOT/scripts/train-ai-mouth-foundation.command"; else echo "Reuse existing Mouth pack: control_models/ai_mouth_foundation-v1.pt"; fi
echo
echo "Learned pack set ready:"
ls -lh "$ROOT/control_models"/*.pt 2>/dev/null || true
echo "Next: ./deep-train-ai-voicebank.command to pin every compatible pack into .yuaz-0.2.8ai12"
