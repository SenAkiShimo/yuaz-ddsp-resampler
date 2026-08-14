#!/bin/bash
exec "$(cd "$(dirname "$0")" && pwd)/scripts/probe-yuaz-checkpoint.command" "$@"
