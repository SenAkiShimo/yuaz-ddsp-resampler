#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -f config.json ]; then
  echo "Run scripts/configure-macos.command first."
  exit 1
fi
DEST="$HOME/Library/OpenUtau/Resamplers"
mkdir -p "$DEST"
chmod +x "$ROOT/yuaz-ddsp-resampler"
NAME="Yuaz-DDSP-Resampler-v0.2.7-alpha.8-rc.3.2.sh"
cat > "$DEST/$NAME" <<SCRIPT
#!/bin/bash
exec "$ROOT/yuaz-ddsp-resampler" "\$@"
SCRIPT
chmod +x "$DEST/$NAME"
MANIFEST="${NAME%.sh}.yaml"
cp "$ROOT/resampler-manifest.yaml" "$DEST/$MANIFEST"
echo "Installed alongside older Yuaz versions: $DEST/$NAME"
echo "Installed expressions: $DEST/$MANIFEST"
echo "Other existing Yuaz resamplers and voicebank adaptation data were preserved."
echo "Restart OpenUtau and select $NAME."
