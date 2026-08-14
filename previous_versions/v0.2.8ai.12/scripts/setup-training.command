#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/scripts/import-existing-control-packs.command"
echo "Yuaz 0.2.8ai.12 Developer Training Setup"
echo "All dataset downloads are resumable. China routes are preferred where useful; official overseas sources may be used with VPN."
echo
echo "  1) ALL remaining packs: VocalSet + Phonation Modes + MOCHA (recommended)"
echo "  2) VocalSet Gender/Formant only (YG)"
echo "  3) Phonation Modes + MOCHA only (YT/YV/YO)"
echo "  4) GTSinger Chinese technique pack (existing YB/YF/YX/YP)"
echo "  5) High-Band Foundation v1 (audit existing GTSinger + VocalSet + Phonation Modes)"
echo "  6) Exit"
read -r -p "Choose [1]: " CHOICE
CHOICE="${CHOICE:-1}"
case "$CHOICE" in
  1) exec "$ROOT/scripts/setup-all-training.command" ;;
  2) exec "$ROOT/scripts/setup-gender-training.command" ;;
  3) exec "$ROOT/scripts/setup-multimodal-training.command" ;;
  4) exec "$ROOT/scripts/setup-ai-training.command" ;;
  5) exec "$ROOT/scripts/audit-highband-datasets.command" ;;
  *) exit 0 ;;
esac
