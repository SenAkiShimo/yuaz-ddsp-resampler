#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -f config.json ]; then
  echo "Run scripts/configure-macos.command first."
  exit 1
fi
"$ROOT/scripts/uninstall-openutau-macos.command"
DEST="$HOME/Library/OpenUtau/Resamplers"
mkdir -p "$DEST"
chmod +x "$ROOT/yuaz-ddsp-resampler"
NAME="Yuaz-DDSP-Resampler-v0.2.7-alpha.1.sh"
cat > "$DEST/$NAME" <<SCRIPT
#!/bin/bash
exec "$ROOT/yuaz-ddsp-resampler" "\$@"
SCRIPT
chmod +x "$DEST/$NAME"
echo "Installed: $DEST/$NAME"
echo "Restart OpenUtau and select $NAME."
