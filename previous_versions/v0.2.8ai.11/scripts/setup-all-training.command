#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Yuaz 0.2.8ai.11 — Download/Resume ALL remaining learned-control data"
echo "Existing VocalSet .part files are reused; nothing complete is downloaded twice."
echo
"$ROOT/scripts/setup-gender-training.command"
echo
"$ROOT/scripts/setup-multimodal-training.command"
