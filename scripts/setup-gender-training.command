#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${YUAZ_CONTROL_DATASETS:-$HOME/YuazControlDatasets}"
DEST="$DATA_ROOT/VocalSetMirror"
TOOLS="$HOME/Library/Application Support/YuazDDSP/training-tools-gender/site-packages"
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
HF_ENDPOINT=""
mkdir -p "$DATA_ROOT" "$TOOLS"
PYTHON="$ROOT/.venv/bin/python"
[ -x "$PYTHON" ] || { echo "Run setup-macos.command first."; exit 1; }

echo "Yuaz 0.2.8ai.13 — Gender/Formant Developer Training Setup"
echo "Developer-only data. Final users can receive the trained .pt inside Yuaz."
echo "Python packages still prefer the Tsinghua PyPI mirror."
echo
echo "Dataset route:"
echo "  1) Official Hugging Face (recommended when VPN is available)"
echo "  2) China mirror hf-mirror.com"
read -r -p "Choose [1]: " ROUTE
ROUTE="${ROUTE:-1}"
case "$ROUTE" in
  1) HF_ENDPOINT="https://huggingface.co"; ROUTE_NAME="Official Hugging Face" ;;
  2) HF_ENDPOINT="https://hf-mirror.com"; ROUTE_NAME="hf-mirror.com" ;;
  *) echo "Invalid route"; exit 1 ;;
esac
echo "  Dataset transport: $HF_ENDPOINT ($ROUTE_NAME)"
echo "  Python packages:    $MIRROR"
echo

PYTHONPATH="$TOOLS${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - <<'PY' >/dev/null 2>&1 || NEED_ARROW=1
import pyarrow
PY
if [ "${NEED_ARROW:-0}" = "1" ]; then
  echo "Installing developer-only pyarrow from Tsinghua PyPI mirror..."
  "$PYTHON" -m pip install --target "$TOOLS" 'pyarrow' -i "$MIRROR"
fi
PYTHONPATH="$TOOLS${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - <<'PY'
import pyarrow
print("pyarrow ready:", pyarrow.__version__)
PY

echo
echo "Checking VocalSet mirror files and exact remaining size..."
"$PYTHON" "$ROOT/scripts/download-vocalset.py" --preset gender-core --local-dir "$DEST" --endpoint "$HF_ENDPOINT" --dry-run
read -r -p "Start this resumable download? [Y/n]: " CONFIRM
CONFIRM="${CONFIRM:-Y}"
case "$CONFIRM" in y|Y|yes|YES) ;; *) echo "Cancelled."; exit 0;; esac
"$PYTHON" "$ROOT/scripts/download-vocalset.py" --preset gender-core --local-dir "$DEST" --endpoint "$HF_ENDPOINT"
printf '%s\n' "$DEST" > "$DATA_ROOT/ACTIVE_VOCALSET_ROOT.txt"
echo
echo "VocalSet developer data ready: $DEST"
echo "Next: ./train-ai-gender-foundation.command"
