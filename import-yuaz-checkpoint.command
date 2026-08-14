#!/bin/bash
exec "$(cd "$(dirname "$0")" && pwd)/scripts/import-yuaz-checkpoint.command" "$@"
