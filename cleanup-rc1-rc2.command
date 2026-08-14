#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/scripts/cleanup-rc1-rc2.command" "$@"
