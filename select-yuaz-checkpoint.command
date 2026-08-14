#!/bin/bash
exec "$(cd "$(dirname "$0")" && pwd)/scripts/select-yuaz-checkpoint.command" "$@"
