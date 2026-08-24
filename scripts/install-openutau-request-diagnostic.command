#!/bin/bash
set -euo pipefail
SOURCE="$(cd "$(dirname "$0")/.." && pwd)"
"$SOURCE/scripts/install-openutau-macos.command"
FINAL="$HOME/Library/Application Support/YuazDDSP/0.2.9"
DEST="$HOME/Library/OpenUtau/Resamplers"
ENTRY="$DEST/Yuaz-DDSP-Resampler-v0.2.9.sh"
cat > "$ENTRY" <<'SCRIPT'
#!/bin/bash
set -u
FINAL="$HOME/Library/Application Support/YuazDDSP/0.2.9"
LOGDIR="$FINAL/logs/openutau-entry"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
LOG="$LOGDIR/$STAMP.args"
{
  printf 'entry_pid=%s\n' "$$"
  printf 'entry_path=%q\n' "$0"
  printf 'argc=%s\n' "$#"
  i=0
  for arg in "$@"; do
    printf 'arg[%d]=' "$i"
    printf '%q' "$arg"
    printf '\n'
    i=$((i + 1))
  done
} > "$LOG"
exec "$FINAL/yuaz-ddsp-resampler" "$@"
SCRIPT
chmod +x "$ENTRY"
cat > "$FINAL/yuaz-ddsp-resampler" <<'SCRIPT'
#!/bin/bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "Yuaz DDSP Resampler: run scripts/setup-macos.command first." >&2
  exit 1
fi
LOGDIR="$ROOT/logs/client-requests"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
ARGS="$LOGDIR/$STAMP.args"
ERR="$LOGDIR/$STAMP.err"
{
  printf 'pid=%s\n' "$$"
  printf 'argc=%s\n' "$#"
  i=0
  for arg in "$@"; do
    printf 'arg[%d]=' "$i"
    printf '%q' "$arg"
    printf '\n'
    i=$((i + 1))
  done
} > "$ARGS"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
set +e
"$ROOT/.venv/bin/python" -m yuaz_ddsp_resampler.client "$@" 2> >(tee "$ERR" >&2)
status=$?
set -e
printf 'exit=%d\n' "$status" >> "$ARGS"
exit "$status"
SCRIPT
chmod +x "$FINAL/yuaz-ddsp-resampler"
cat > "$FINAL/scripts/show-openutau-request-diagnostic.command" <<'SCRIPT'
#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
show_recent() {
  local title="$1"
  local dir="$2"
  echo "=== $title ==="
  if [ ! -d "$dir" ]; then
    echo '(none)'
    return
  fi
  files="$(find "$dir" -type f -name '*.args' -print0 | xargs -0 ls -1t 2>/dev/null | head -n 6 || true)"
  if [ -z "$files" ]; then
    echo '(none)'
    return
  fi
  while IFS= read -r file; do
    [ -n "$file" ] || continue
    echo
    echo "--- $file ---"
    cat "$file"
    err="${file%.args}.err"
    if [ -s "$err" ]; then
      echo 'stderr:'
      cat "$err"
    fi
  done <<< "$files"
}
show_recent 'OpenUtau entry calls' "$ROOT/logs/openutau-entry"
echo
show_recent 'client requests' "$ROOT/logs/client-requests"
echo
echo '=== engine log ==='
tail -n 160 "$ROOT/logs/engine.log" 2>/dev/null || true
SCRIPT
chmod +x "$FINAL/scripts/show-openutau-request-diagnostic.command"
rm -rf "$FINAL/logs/openutau-entry" "$FINAL/logs/client-requests"
mkdir -p "$FINAL/logs/openutau-entry" "$FINAL/logs/client-requests"
pkill -f yuaz_ddsp_resampler.server 2>/dev/null || true
echo "Request diagnostic installed"
echo "$FINAL/scripts/show-openutau-request-diagnostic.command"
