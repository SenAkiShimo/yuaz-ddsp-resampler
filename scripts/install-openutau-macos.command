#!/bin/bash
set -euo pipefail
SOURCE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SOURCE"
[ -f config.json ] || { echo "Run configure-macos.command first."; exit 1; }
[ -x .venv/bin/python ] || { echo "Run setup-macos.command first."; exit 1; }
APP="$HOME/Library/Application Support/YuazDDSP"
FINAL="$APP/0.2.8ai.14"
TMP="$APP/.0.2.8ai.14-installing-$$"
DEST="$HOME/Library/OpenUtau/Resamplers"
NAME="Yuaz-DDSP-Resampler-v0.2.8ai.14.sh"
mkdir -p "$APP" "$DEST"
# ai.13 is intentionally untouched: no stop, no delete, no wrapper replacement, no state migration.
rm -rf "$TMP"; mkdir -p "$TMP"
rsync -a --delete --exclude '.venv' --exclude 'logs' --exclude '.engine-start.lock' --exclude 'engine.pid' "$SOURCE/" "$TMP/"
ENVREAL="$(cd "$SOURCE/.venv" && pwd -P)"; ln -s "$ENVREAL" "$TMP/.venv"
"$TMP/scripts/self-test.command"
python3 - "$TMP/config.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); assert c['engine_version']=='0.2.8ai.14'; assert c['port']==47886
assert c['runtime_id']=='yuaz-0.2.8ai.14-control-v14'; assert c['state_namespace']=='.yuaz-0.2.8ai14'; assert c['preserve_ai13'] is True
print('Runtime config identity OK')
PY
if [ -x "$FINAL/scripts/stop-engine.command" ]; then "$FINAL/scripts/stop-engine.command" 2>/dev/null || true; fi
rm -rf "$FINAL"; mv "$TMP" "$FINAL"
cat > "$DEST/$NAME" <<SCRIPT
#!/bin/bash
exec "$FINAL/yuaz-ddsp-resampler" "\$@"
SCRIPT
chmod +x "$DEST/$NAME"; cp "$FINAL/resampler-manifest.yaml" "$DEST/${NAME%.sh}.yaml"
echo "Installed ai.14 runtime: $FINAL"
echo "Installed ai.14 OpenUtau resampler: $DEST/$NAME"
echo "PRESERVED: ai.13 runtime, ai.13 OpenUtau wrapper/YAML, and every .yuaz-0.2.8ai13 voicebank state."
echo "No purge or predecessor migration was performed."
