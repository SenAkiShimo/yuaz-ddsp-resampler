#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)";exec "$ROOT/scripts/train-all-learned-packs.command" "$@"
