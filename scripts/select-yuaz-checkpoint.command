#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m yuaz_ddsp_resampler.checkpoint_registry list
echo "Enter the model-id prefix to activate:"; read -r ID
.venv/bin/python -m yuaz_ddsp_resampler.checkpoint_registry select "$ID" --project-root "$ROOT"
"$ROOT/scripts/stop-engine.command" 2>/dev/null || true
echo "Selected. Existing ai.14 generations are preserved; incompatible base-SHA generations will be rejected until a matching Deep generation is active."
