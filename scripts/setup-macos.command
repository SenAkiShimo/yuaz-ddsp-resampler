#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOCKFILE="$ROOT/requirements.lock.txt"
[ -f "$LOCKFILE" ] || { echo "requirements.lock.txt not found."; exit 1; }
HASH="$(shasum -a 256 "$LOCKFILE" | awk '{print substr($1,1,16)}')"
APP="$HOME/Library/Application Support/YuazDDSP"
ENVROOT="$APP/environments/0.2.8ai.14-$HASH"
PREV_ENV13="$APP/environments/0.2.8ai.13-$HASH"
PREV_ENV12="$APP/environments/0.2.8ai.12-$HASH"
PREV_ENV11="$APP/environments/0.2.8ai.11-$HASH"
PREV_ENV10="$APP/environments/0.2.8ai.10-$HASH"
MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
mkdir -p "$APP/environments"
verify_env() {
  local ENV="$1"
  [ -x "$ENV/bin/python" ] || return 1
  "$ENV/bin/python" - "$HASH" <<'PY'
import json, platform, sys
from pathlib import Path
expected={
 'torch':'2.13.0','numpy':'2.4.6','librosa':'0.11.0','soundfile':'0.14.0','pyyaml':'6.0.3'
}
try:
 import torch,numpy,librosa,soundfile,yaml
 got={'torch':torch.__version__.split('+')[0],'numpy':numpy.__version__,'librosa':librosa.__version__,'soundfile':soundfile.__version__,'pyyaml':yaml.__version__}
 assert got==expected, (got,expected)
 marker=Path(sys.prefix)/'RUNTIME_ENVIRONMENT.json'
 if marker.exists():
  m=json.loads(marker.read_text(encoding='utf-8'))
  assert m.get('requirements_hash')==sys.argv[1]
  assert m.get('python_version')==platform.python_version()
 print('Pinned environment OK:', platform.python_version(), got)
except Exception as exc:
 print('Environment verification failed:',exc,file=sys.stderr)
 raise SystemExit(1)
PY
}
if [ ! -x "$ENVROOT/bin/python" ] && [ -x "$PREV_ENV13/bin/python" ] && verify_env "$PREV_ENV13"; then
  ENVROOT="$PREV_ENV13"
  echo "Reusing compatible 0.2.8ai.13 pinned environment: $ENVROOT"
elif [ ! -x "$ENVROOT/bin/python" ] && [ -x "$PREV_ENV12/bin/python" ] && verify_env "$PREV_ENV12"; then
  ENVROOT="$PREV_ENV12"
  echo "Reusing compatible 0.2.8ai.12 pinned environment: $ENVROOT"
elif [ ! -x "$ENVROOT/bin/python" ] && [ -x "$PREV_ENV11/bin/python" ] && verify_env "$PREV_ENV11"; then
  ENVROOT="$PREV_ENV11"
  echo "Reusing compatible 0.2.8ai.11 pinned environment: $ENVROOT"
elif [ ! -x "$ENVROOT/bin/python" ] && [ -x "$PREV_ENV10/bin/python" ] && verify_env "$PREV_ENV10"; then
  ENVROOT="$PREV_ENV10"
  echo "Reusing compatible 0.2.8ai.10 pinned environment: $ENVROOT"
fi
if [ -d "$ENVROOT" ] && ! verify_env "$ENVROOT"; then
  BAD="$ENVROOT.invalid-$(date +%Y%m%d-%H%M%S)"
  mv "$ENVROOT" "$BAD"
  echo "Quarantined invalid shared environment: $BAD"
fi
if [ ! -x "$ENVROOT/bin/python" ]; then
  TMP="$ENVROOT.installing.$$"
  rm -rf "$TMP"
  python3 -m venv "$TMP"
  "$TMP/bin/python" -m pip install 'pip==26.2.1' -i "$MIRROR"
  "$TMP/bin/python" -m pip install -r "$LOCKFILE" -i "$MIRROR"
  "$TMP/bin/python" - "$HASH" <<'PY'
import json,platform,sys
from pathlib import Path
import torch,numpy,librosa,soundfile,yaml
expected={'torch':'2.13.0','numpy':'2.4.6','librosa':'0.11.0','soundfile':'0.14.0','pyyaml':'6.0.3'}
got={'torch':torch.__version__.split('+')[0],'numpy':numpy.__version__,'librosa':librosa.__version__,'soundfile':soundfile.__version__,'pyyaml':yaml.__version__}
assert got==expected,(got,expected)
marker={'format':1,'requirements_hash':sys.argv[1],'python_version':platform.python_version(),'packages':got}
(Path(sys.prefix)/'RUNTIME_ENVIRONMENT.json').write_text(json.dumps(marker,indent=2),encoding='utf-8')
print('Built pinned environment:',marker)
PY
  mv "$TMP" "$ENVROOT"
  verify_env "$ENVROOT" || { echo "Pinned environment failed verification after installation."; exit 1; }
else
  echo "Pinned shared environment already exists: $ENVROOT"
fi
rm -rf "$ROOT/.venv"
ln -s "$ENVROOT" "$ROOT/.venv"
echo "Environment linked: $ROOT/.venv -> $ENVROOT"
