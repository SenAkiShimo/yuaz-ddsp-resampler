#!/bin/bash
set -euo pipefail
SOURCE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SOURCE"
[ -f config.json ] || { echo "Run scripts/configure-macos.command first."; exit 1; }
[ -x .venv/bin/python ] || { echo "Run scripts/setup-macos.command first."; exit 1; }

APP="$HOME/Library/Application Support/YuazDDSP"
FINAL="$APP/0.2.8ai.16-yf-yb-test"
TMP="$APP/.0.2.8ai.16-yf-yb-test-installing-$$"
DEST="$HOME/Library/OpenUtau/Resamplers"
NAME="Yuaz-DDSP-Resampler-YF-YB-Test-R2.sh"
OLD_NAME="Yuaz-DDSP-Resampler-YF-YB-Test.sh"

python3 - "$SOURCE/config.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
assert c['engine_version']=='0.2.8ai.16'
assert c['port']==47889
assert c['runtime_id']=='yuaz-0.2.8ai.16-yf-yb-test'
assert c['state_namespace']=='.yuaz-0.2.8ai14'
assert c['state_access']=='read-only-ai14-compatibility'
assert c['preserve_ai14'] is True
assert c['allow_ai16_voicebank_training'] is False
print('Test runtime config OK')
PY

mkdir -p "$APP" "$DEST"
if [ -x "$FINAL/scripts/stop-engine.command" ]; then
  "$FINAL/scripts/stop-engine.command" 2>/dev/null || true
fi
rm -rf "$TMP"
mkdir -p "$TMP"
rsync -a --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'logs' \
  --exclude '.engine-start.lock' \
  --exclude 'engine.pid' \
  --exclude 'yf-yb-listen-test-output' \
  "$SOURCE/" "$TMP/"
ENVREAL="$(cd "$SOURCE/.venv" && pwd -P)"
ln -s "$ENVREAL" "$TMP/.venv"

PYTHONPATH="$TMP/src" "$TMP/.venv/bin/python" -m compileall -q "$TMP/src"
python3 - "$TMP/config.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
assert c['port']==47889
assert c['runtime_id']=='yuaz-0.2.8ai.16-yf-yb-test'
print('Installed runtime identity OK')
PY

rm -rf "$FINAL"
mv "$TMP" "$FINAL"
mkdir -p "$FINAL/logs"
LOG="$FINAL/logs/client.log"
printf '[%s] installed wrapper=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$DEST/$NAME" > "$LOG"

rm -f "$DEST/$OLD_NAME" "$DEST/${OLD_NAME%.sh}.yaml"
cat > "$DEST/$NAME" <<SCRIPT
#!/bin/bash
LOG="$FINAL/logs/client.log"
{
  printf '\n[%s] OpenUtau invocation\n' "\$(date '+%Y-%m-%d %H:%M:%S')"
  printf 'wrapper=%q\n' "\$0"
  printf 'pwd=%q\n' "\$PWD"
  printf 'argc=%d\n' "\$#"
  i=0
  for arg in "\$@"; do
    printf 'arg[%d]=%q\n' "\$i" "\$arg"
    i=\$((i+1))
  done
} >> "\$LOG"
"$FINAL/yuaz-ddsp-resampler" "\$@" 2>> "\$LOG"
status=\$?
printf 'exit=%d\n' "\$status" >> "\$LOG"
exit "\$status"
SCRIPT
chmod +x "$DEST/$NAME"
cp "$FINAL/resampler-manifest.yaml" "$DEST/${NAME%.sh}.yaml"

echo "Installed test runtime: $FINAL"
echo "OpenUtau resampler: $DEST/$NAME"
echo "Client log: $LOG"
echo "Port: 47889"
echo "PRESERVED: $APP/0.2.8ai.16"
echo "PRESERVED: .yuaz-0.2.8ai14"
