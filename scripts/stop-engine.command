#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
pkill -f "$ROOT/config.json" 2>/dev/null || true
rm -f "$ROOT/.engine-start.lock"
echo "Engine stopped."
