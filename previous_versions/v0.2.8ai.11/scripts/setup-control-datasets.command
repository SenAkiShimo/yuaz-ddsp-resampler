#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "This command is now named setup-ai-training.command."
exec "$ROOT/scripts/setup-ai-training.command" "$@"
