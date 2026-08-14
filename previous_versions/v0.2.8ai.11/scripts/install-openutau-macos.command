#!/bin/bash
set -euo pipefail
SOURCE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SOURCE"
[ -f config.json ] || { echo "Run configure-macos.command first."; exit 1; }
[ -x .venv/bin/python ] || { echo "Run setup-macos.command first."; exit 1; }
APP="$HOME/Library/Application Support/YuazDDSP"
FINAL="$APP/0.2.8ai.11"
TMP="$APP/.0.2.8ai.11-installing-$$"
DEST="$HOME/Library/OpenUtau/Resamplers"
NAME="Yuaz-DDSP-Resampler-v0.2.8ai.11.sh"
mkdir -p "$APP" "$DEST"

# Preserve the currently working generation before replacing installed versions.
"$SOURCE/scripts/backup-current-stable.command"

if [ -x "$FINAL/scripts/stop-engine.command" ]; then
  "$FINAL/scripts/stop-engine.command" 2>/dev/null || true
fi
rm -rf "$TMP"
mkdir -p "$TMP"
rsync -a --delete \
  --exclude '.venv' --exclude 'logs' --exclude '.engine-start.lock' --exclude 'engine.pid' \
  "$SOURCE/" "$TMP/"
ENVREAL="$(cd "$SOURCE/.venv" && pwd -P)"
ln -s "$ENVREAL" "$TMP/.venv"

"$TMP/scripts/self-test.command"
python3 - "$TMP/config.json" <<'PY2'
import json,sys
c=json.load(open(sys.argv[1]))
assert c['engine_version']=='0.2.8ai.11'
assert c['port']==47885
assert c['runtime_id']=='yuaz-0.2.8ai.11-control-v11'
print('Runtime config identity OK')
PY2
rm -rf "$FINAL"
mv "$TMP" "$FINAL"
cat > "$DEST/$NAME" <<SCRIPT
#!/bin/bash
exec "$FINAL/yuaz-ddsp-resampler" "\$@"
SCRIPT
chmod +x "$DEST/$NAME"
cp "$FINAL/resampler-manifest.yaml" "$DEST/${NAME%.sh}.yaml"

echo "Installed Yuaz runtime: $FINAL"
echo "Installed OpenUtau resampler: $DEST/$NAME"
echo "Migrating prepared voicebank state and removing previous installed Yuaz versions..."
"$FINAL/scripts/migrate-and-purge-previous.command"
echo "0.2.8ai.11 is now the only installed Yuaz resampler version."
