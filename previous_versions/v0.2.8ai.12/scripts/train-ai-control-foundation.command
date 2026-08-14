#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -x .venv/bin/python ] || { echo "Run setup-macos.command first."; exit 1; }
strip_path() { python3 - "$1" <<'PY2'
import shlex,sys
parts=shlex.split(sys.argv[1].strip()); print(parts[0] if parts else '')
PY2
}
DATA_ROOT="${YUAZ_CONTROL_DATASETS:-$HOME/YuazControlDatasets}"
CACHE="$DATA_ROOT/_yuaz_ai_cache/gtsinger-ddsp-v2-direct"
OUT="$ROOT/control_models/ai_control_foundation-v2.pt"
MARKER="$DATA_ROOT/ACTIVE_GTSINGER_ROOT.txt"
mkdir -p "$CACHE" "$ROOT/control_models"
cat <<EOF
Yuaz AI Control Foundation v2 Trainer

Direct supervised controls in this branch:
  YB Breathiness
  YF Falsetto
  YX Mixed Voice
  YP Pharyngeal

YT Tension and YV Voicing are deliberately NOT trained from GTSinger because
GTSinger does not label them directly. They keep the deterministic RC4.2 DDSP path.

Both sides of every paired technique example are analyzed by the frozen current
Yuaz encoder + DDSP decoder. The small neural controller learns residuals only in
Yuaz-native spectral-envelope / AP / harmonic-gate space.
EOF
if [ -f "$MARKER" ] && [ -d "$(cat "$MARKER")" ]; then
  GTS="$(cat "$MARKER")"
  echo "Using developer dataset configured by setup-ai-training.command:"
  echo "  $GTS"
else
  echo "No configured GTSinger root found."
  echo "Recommended: run ./setup-ai-training.command first."
  read -r -p "Or drop an existing GTSinger root here: " RAW
  GTS="$(strip_path "$RAW")"
fi
[ -d "$GTS" ] || { echo "GTSinger folder not found: $GTS"; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.ai_control_training coverage "$GTS"
"$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.ai_control_training build-gtsinger "$GTS" "$CACHE" --project-root "$ROOT" --feature-backend yuaz-native
EPOCHS="${YUAZ_AI_CONTROL_EPOCHS:-12}"
"$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.ai_control_training train "$CACHE" "$OUT" --epochs "$EPOCHS"
echo "AI Control Foundation v2 trained: $OUT"
echo "Now run deep-train-ai-voicebank.command so a frozen copy is pinned into the isolated AI generation."
