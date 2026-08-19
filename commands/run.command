#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$ROOT/scripts"

list_names() {
  find "$SCRIPTS" -maxdepth 1 -type f -name '*.command' -print \
    | sed 's#.*/##; s/\.command$//' \
    | LC_ALL=C sort
}

print_group() {
  local title="$1"
  local pattern="$2"
  local names
  names="$(list_names | grep -E "$pattern" || true)"
  if [ -n "$names" ]; then
    echo
    echo "$title"
    printf '  %s\n' $names
  fi
}

show_help() {
  echo "Yuaz command launcher"
  echo
  echo "Usage:"
  echo "  ./commands/run.command <name> [arguments...]"
  echo "  ./commands/run.command list"
  echo "  ./commands/run.command find <text>"
  echo
  echo "Examples:"
  echo "  ./commands/run.command doctor"
  echo "  ./commands/run.command self-test"
  echo "  ./commands/run.command probe-yv-final-chain"
  echo "  ./commands/run.command train-ai-control-foundation"
  echo
  echo "Implementations live in scripts/. This launcher is the public command entry point."

  print_group "Setup / install" '^(setup-|configure-|install-|uninstall-|doctor$|self-test$|prepare-voicebank$)'
  print_group "Training / preparation" '^(train-|learn-|prepare-|repin-|download-)'
  print_group "Diagnostics / audit" '^(probe-|audit-|diagnose-|compare-|.*-test$|.*diagnostic$)'
  print_group "Maintenance / state" '^(backup-|restore-|rollback-|reset-|cleanup-|purge-|remove-|migrate-|finalize-|stop-engine$|list-|select-|import-)'
}

if [ ! -d "$SCRIPTS" ]; then
  echo "Scripts directory not found: $SCRIPTS" >&2
  exit 1
fi

name="${1:-}"
case "$name" in
  ""|help|-h|--help)
    show_help
    exit 0
    ;;
  list)
    list_names
    exit 0
    ;;
  find)
    shift
    query="${1:-}"
    if [ -z "$query" ]; then
      echo "Usage: ./commands/run.command find <text>" >&2
      exit 2
    fi
    list_names | grep -i -- "$query" || true
    exit 0
    ;;
esac

shift
name="${name%.command}"

case "$name" in
  deep-train-ai-voicebank)
    name="deep-train-voicebank"
    ;;
  highband-nyquist-diagnostic)
    name="highband-test"
    ;;
esac

target="$SCRIPTS/$name.command"
if [ ! -f "$target" ]; then
  echo "Unknown Yuaz command: $name" >&2
  echo >&2
  echo "Matching commands:" >&2
  matches="$(list_names | grep -i -- "$name" || true)"
  if [ -n "$matches" ]; then
    printf '  %s\n' $matches >&2
  else
    echo "  (none)" >&2
  fi
  echo >&2
  echo "Run ./commands/run.command list to see every command." >&2
  exit 2
fi

exec /bin/bash "$target" "$@"
