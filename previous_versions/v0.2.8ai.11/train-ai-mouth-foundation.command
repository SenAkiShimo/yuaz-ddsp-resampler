#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)";exec "$ROOT/scripts/train-ai-mouth-foundation.command" "$@"
