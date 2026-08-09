#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ -x .venv/bin/python ]; then
  echo "Environment already exists: $ROOT/.venv"
  exit 0
fi
python3 -m venv .venv
source .venv/bin/activate
MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
python -m pip install -U pip -i "$MIRROR" || python -m pip install -U pip
python -m pip install -r requirements.txt -i "$MIRROR" || python -m pip install -r requirements.txt
python - <<'PY'
import torch, librosa, soundfile, yaml, numpy
print("Environment OK")
PY
