#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "Run setup-macos.command first."; exit 1; }
DATA_ROOT="${YUAZ_CONTROL_DATASETS:-$HOME/YuazControlDatasets}"
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
CACHE="$DATA_ROOT/_yuaz_ai_cache/phonation-osf-mocha-v1-$SHORT"
DIRECT="$CACHE/phonation_direct_shards"
MOCHA="$CACHE/phonation_shards"
OUT="$ROOT/control_models/ai_phonation_foundation-v1-signedfix-$SHORT.pt"
[ -d "$DIRECT" ] || { echo "Missing current-checkpoint phonation shards: $DIRECT"; exit 1; }
[ -d "$MOCHA" ] || { echo "Missing current-checkpoint MOCHA voicing shards: $MOCHA"; exit 1; }
mkdir -p "$ROOT/control_models"
echo "Yuaz Phonation Signedfix"
echo "Checkpoint identity: $SHA"
echo "Direct shards: $DIRECT"
echo "MOCHA shards: $MOCHA"
echo "Output: $OUT"
EPOCHS="${YUAZ_AI_PHONATION_SIGNEDFIX_EPOCHS:-12}"
"$PY" -m yuaz_ddsp_resampler.phonation_signed_training "$DIRECT" "$MOCHA" "$OUT" --checkpoint-sha "$SHA" --epochs "$EPOCHS"
"$PY" - "$OUT" "$SHA" <<'PY'
import sys
from yuaz_ddsp_resampler.ai_vocal_controls import load_ai_control_adapter
p,sha=sys.argv[1:]
pack,meta=load_ai_control_adapter(p,device='cpu')
assert tuple(pack.control_names)==('tension','voicing')
assert tuple(pack.control_modes)==('signed','signed')
assert tuple(pack.output_scopes)==('spectral','ap','gate')
assert str(meta.get('feature_backend') or '')=='yuaz-native-ddsp-v1'
assert str(meta.get('checkpoint_sha256') or '')==sha
assert str(meta.get('training_method') or '')=='phonation signedfix v2'
print('Signedfix provenance verified:', sha)
print('Best validation loss:', meta.get('best_validation_loss'))
print('Saved candidate only; canonical phonation foundation is unchanged.')
PY
