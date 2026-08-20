#!/bin/bash
set -euo pipefail

APP="$HOME/Library/Application Support/YuazDDSP"
DEST="$HOME/Library/OpenUtau/Resamplers"
KEEP_RUNTIME="0.2.9"
KEEP_WRAPPER="Yuaz-DDSP-Resampler-v0.2.9"

stop_runtime_dir() {
  local dir="$1"
  [ -d "$dir" ] || return 0
  if [ -x "$dir/scripts/stop-engine.command" ]; then
    "$dir/scripts/stop-engine.command" >/dev/null 2>&1 || true
  fi
  if [ -f "$dir/engine.pid" ]; then
    local pid
    pid="$(python3 - "$dir/engine.pid" <<'PY'
import json,sys
try:
    print(int(json.load(open(sys.argv[1])).get('pid', 0)))
except Exception:
    print(0)
PY
)"
    if [ "$pid" -gt 0 ] 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  fi
}

echo "Cleaning legacy Yuaz resampler installations..."

if [ -d "$APP" ]; then
  while IFS= read -r -d '' dir; do
    name="$(basename "$dir")"
    case "$name" in
      "$KEEP_RUNTIME"|state|environments|checkpoints|models)
        continue
        ;;
      .*)
        continue
        ;;
    esac
    case "$name" in
      0.*|v0.*|yuaz-*|Yuaz-*)
        echo "Removing runtime: $dir"
        stop_runtime_dir "$dir"
        rm -rf "$dir"
        ;;
    esac
  done < <(find "$APP" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
fi

# Stop test/development servers whose config is outside the retained 0.2.9 runtime.
while IFS= read -r line; do
  pid="${line%% *}"
  cmd="${line#* }"
  case "$cmd" in
    *yuaz_ddsp_resampler.server*--config*)
      case "$cmd" in
        *"$APP/$KEEP_RUNTIME/config.json"*) ;;
        *)
          echo "Stopping legacy Yuaz server PID $pid"
          kill "$pid" 2>/dev/null || true
          ;;
      esac
      ;;
  esac
done < <(ps -axo pid=,command= 2>/dev/null | sed 's/^[[:space:]]*//' || true)

if [ -d "$DEST" ]; then
  while IFS= read -r -d '' file; do
    base="$(basename "$file")"
    case "$base" in
      "$KEEP_WRAPPER.sh"|"$KEEP_WRAPPER.yaml")
        continue
        ;;
      Yuaz-DDSP-Resampler*.sh|Yuaz-DDSP-Resampler*.yaml|Yuaz*DDSP*Resampler*.sh|Yuaz*DDSP*Resampler*.yaml)
        echo "Removing OpenUtau wrapper: $file"
        rm -f "$file"
        ;;
    esac
  done < <(find "$DEST" -mindepth 1 -maxdepth 1 -type f -print0 2>/dev/null)
fi

echo
echo "Cleanup complete."
echo "PRESERVED runtime: $APP/$KEEP_RUNTIME"
echo "PRESERVED wrapper: $DEST/$KEEP_WRAPPER.sh"
echo "PRESERVED: voicebank .yuaz-* state directories"
echo "PRESERVED: checkpoints, shared environments, and source repositories"
