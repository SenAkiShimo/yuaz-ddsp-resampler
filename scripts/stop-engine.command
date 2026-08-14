#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE="$ROOT/engine.pid"
if [ -f "$PIDFILE" ]; then
  PID="$(python3 - "$PIDFILE" <<'PY'
import json,sys
try: print(int(json.load(open(sys.argv[1]))['pid']))
except Exception: print('')
PY
)"
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$PID" 2>/dev/null || break
      sleep 0.1
    done
    kill -9 "$PID" 2>/dev/null || true
  fi
fi
pkill -f "yuaz_ddsp_resampler.server --config $ROOT/config.json" 2>/dev/null || true
rm -f "$ROOT/.engine-start.lock" "$ROOT/engine.pid"
echo "0.2.8ai.13 engine stopped: $ROOT"
