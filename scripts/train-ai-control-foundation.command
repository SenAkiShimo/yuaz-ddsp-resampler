#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "Run setup-macos.command first."; exit 1; }
strip_path() { python3 - "$1" <<'PY2'
import shlex,sys
parts=shlex.split(sys.argv[1].strip()); print(parts[0] if parts else '')
PY2
}
DATA_ROOT="${YUAZ_CONTROL_DATASETS:-$HOME/YuazControlDatasets}"
MARKER="$DATA_ROOT/ACTIVE_GTSINGER_ROOT.txt"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
SHA="$($PY - <<'PY'
import json
from pathlib import Path
from yuaz_ddsp_resampler.checkpoint_identity import checkpoint_identity_sha
c=json.loads(Path('config.json').read_text())
print(checkpoint_identity_sha(Path(c['checkpoint']).expanduser()))
PY
)"
SHORT="${SHA:0:16}"
CACHE="$DATA_ROOT/_yuaz_ai_cache/gtsinger-ddsp-v2-direct-$SHORT"
WORK="$ROOT/control_models/ai_control_foundation-v2-$SHORT.pt"
OUT="$ROOT/control_models/ai_control_foundation-v2.pt"
mkdir -p "$CACHE" "$ROOT/control_models"
if [ -f "$MARKER" ] && [ -d "$(cat "$MARKER")" ]; then
  GTS="$(cat "$MARKER")"
else
  read -r -p "Drop an existing GTSinger root here: " RAW
  GTS="$(strip_path "$RAW")"
fi
[ -d "$GTS" ] || { echo "GTSinger folder not found: $GTS"; exit 1; }
echo "Yuaz technique foundation"
echo "Checkpoint identity: $SHA"
echo "Cache: $CACHE"
"$PY" -m yuaz_ddsp_resampler.ai_control_training coverage "$GTS"
"$PY" -m yuaz_ddsp_resampler.ai_control_training build-gtsinger "$GTS" "$CACHE" --project-root "$ROOT" --feature-backend yuaz-native
EPOCHS="${YUAZ_AI_CONTROL_EPOCHS:-12}"
"$PY" -m yuaz_ddsp_resampler.ai_control_training train "$CACHE" "$WORK" --epochs "$EPOCHS"
"$PY" - "$WORK" "$SHA" <<'PY'
import sys
from yuaz_ddsp_resampler.ai_vocal_controls import load_ai_control_adapter
p,sha=sys.argv[1:]
pack,meta=load_ai_control_adapter(p,device='cpu')
assert tuple(pack.control_names)==('breathiness','falsetto','mixed_voice','pharyngeal')
assert str(meta.get('feature_backend') or '')=='yuaz-native-ddsp-v1'
assert str(meta.get('checkpoint_sha256') or '')==sha
print('Foundation provenance verified:', sha)
PY
STAMP="$(date +%Y%m%d-%H%M%S)"
B="$HOME/Documents/Yuaz-DDSP-Backups/control-models"
mkdir -p "$B"
[ ! -f "$OUT" ] || cp "$OUT" "$B/ai_control_foundation-v2-before-$STAMP.pt"
cp "$WORK" "$OUT"
[ -f "$WORK.json" ] && cp "$WORK.json" "$OUT.json" || true
cp "$OUT" "$B/ai_control_foundation-v2-GTSinger-$SHORT.pt"
[ -f "$OUT.json" ] && cp "$OUT.json" "$B/ai_control_foundation-v2-GTSinger-$SHORT.pt.json" || true
echo "Technique foundation ready: $OUT"
