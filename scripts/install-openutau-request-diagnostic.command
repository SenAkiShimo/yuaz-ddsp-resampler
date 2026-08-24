#!/bin/bash
set -euo pipefail
SOURCE="$(cd "$(dirname "$0")/.." && pwd)"
"$SOURCE/scripts/install-openutau-macos.command"
FINAL="$HOME/Library/Application Support/YuazDDSP/0.2.9"
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
LOGDIR="$ROOT/logs/client-requests"
echo '=== latest client requests ==='
if [ -d "$LOGDIR" ]; then
  find "$LOGDIR" -type f -name '*.args' -print0 \
    | xargs -0 ls -1t 2>/dev/null \
    | head -n 6 \
    | while IFS= read -r file; do
        echo
        echo "--- $file ---"
        cat "$file"
        err="${file%.args}.err"
        if [ -s "$err" ]; then
          echo 'stderr:'
          cat "$err"
        fi
      done
else
  echo '(none)'
fi
echo
echo '=== engine log ==='
tail -n 160 "$ROOT/logs/engine.log" 2>/dev/null || true
SCRIPT
chmod +x "$FINAL/scripts/show-openutau-request-diagnostic.command"
rm -rf "$FINAL/logs/client-requests"
mkdir -p "$FINAL/logs/client-requests"
pkill -f yuaz_ddsp_resampler.server 2>/dev/null || true
echo "Request diagnostic installed"
echo "$FINAL/scripts/show-openutau-request-diagnostic.command"
