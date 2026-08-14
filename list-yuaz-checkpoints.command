#!/bin/bash
exec "$(cd "$(dirname "$0")" && pwd)/scripts/list-yuaz-checkpoints.command" "$@"
