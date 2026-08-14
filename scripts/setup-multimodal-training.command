#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${YUAZ_CONTROL_DATASETS:-$HOME/YuazControlDatasets}"
PHONATION="$DATA_ROOT/PhonationModesOSF"
MOCHA="$DATA_ROOT/MOCHA-TIMIT"
PYTHON="$ROOT/.venv/bin/python"
[ -x "$PYTHON" ] || { echo "Run setup-macos.command first."; exit 1; }

echo "Yuaz 0.2.8ai.14 — Tension / Voicing / Mouth Developer Data"
echo "All downloads are resumable (.part + HTTP Range). Official overseas sources may be used with VPN."
echo
echo "OSF Phonation Modes — YT Tension (breathy / neutral-modal / pressed singing)"
echo "MOCHA-TIMIT CSTR official — YV Voicing/Closure + YO Mouth (laryngograph + EMA)"
echo "The invalid 62 KiB PHAIDRA object from 0.2.8ai.4 is ignored and is NOT treated as training data."
echo
"$PYTHON" "$ROOT/scripts/download-phonation-modes.py" --local-dir "$PHONATION" --dry-run
"$PYTHON" "$ROOT/scripts/download-mocha.py" --local-dir "$MOCHA" --dry-run
read -r -p "Download/resume Phonation Modes + MOCHA now? [Y/n]: " C
C="${C:-Y}"
case "$C" in y|Y|yes|YES) ;; *) echo "Cancelled."; exit 0;; esac
"$PYTHON" "$ROOT/scripts/download-phonation-modes.py" --local-dir "$PHONATION"
"$PYTHON" "$ROOT/scripts/download-mocha.py" --local-dir "$MOCHA"
printf '%s\n' "$PHONATION" > "$DATA_ROOT/ACTIVE_PHONATION_ROOT.txt"
printf '%s\n' "$MOCHA" > "$DATA_ROOT/ACTIVE_MOCHA_ROOT.txt"
echo
echo "Multimodal developer data ready."
echo "Next: ./train-all-learned-packs.command"
