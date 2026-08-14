#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"; exec "$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.checkpoint_registry list
