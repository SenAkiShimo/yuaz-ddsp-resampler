#!/bin/bash
set -euo pipefail
SOURCE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SOURCE"
[ -f config.json ] || { echo "Run configure-macos.command first."; exit 1; }
[ -x .venv/bin/python ] || { echo "Run setup-macos.command first."; exit 1; }
APP="$HOME/Library/Application Support/YuazDDSP"
FINAL="$APP/0.2.8ai.15"
TMP="$APP/.0.2.8ai.15-installing-$$"
DEST="$HOME/Library/OpenUtau/Resamplers"
NAME="Yuaz-DDSP-Resampler-v0.2.8ai.15.sh"
mkdir -p "$APP" "$DEST"
"$SOURCE/scripts/purge-previous-version.command"
rm -rf "$TMP"; mkdir -p "$TMP"
rsync -a --delete --exclude '.venv' --exclude 'logs' --exclude '.engine-start.lock' --exclude 'engine.pid' "$SOURCE/" "$TMP/"
ENVREAL="$(cd "$SOURCE/.venv" && pwd -P)"; ln -s "$ENVREAL" "$TMP/.venv"
"$TMP/scripts/self-test.command"
python3 - "$TMP/config.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
assert c['engine_version']=='0.2.8ai.15'
assert c['port']==47887
assert c['runtime_id']=='yuaz-0.2.8ai.15-control-calibration-v15'
assert c['state_namespace']=='.yuaz-0.2.8ai14'
assert c['state_access']=='read-only-ai14-compatibility'
assert c['preserve_ai14'] is True
assert c['allow_ai15_voicebank_training'] is False
print('Runtime config identity OK')
PY
if [ -x "$FINAL/scripts/stop-engine.command" ]; then "$FINAL/scripts/stop-engine.command" 2>/dev/null || true; fi
rm -rf "$FINAL"; mv "$TMP" "$FINAL"
cat > "$DEST/$NAME" <<SCRIPT
#!/bin/bash
exec "$FINAL/yuaz-ddsp-resampler" "\$@"
SCRIPT
chmod +x "$DEST/$NAME"
cp "$FINAL/resampler-manifest.yaml" "$DEST/${NAME%.sh}.yaml"
echo "Installed: $FINAL"
echo "Resampler: $DEST/$NAME"
echo "PRESERVED: .yuaz-0.2.8ai14"
