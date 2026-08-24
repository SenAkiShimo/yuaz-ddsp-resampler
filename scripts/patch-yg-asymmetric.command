#!/bin/bash
set -euo pipefail
RUNTIME="$HOME/Library/Application Support/YuazDDSP/0.2.9"
TARGET="$RUNTIME/src/yuaz_ddsp_resampler/post_gender.py"
python3 - "$TARGET" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
old = '''    a = abs(amount)\n    semitones = np.sign(amount) * (20.0 * a + 14.0 * (a ** 3))\n'''
new = '''    a = abs(amount)\n    if amount >= 0.0:\n        semitones = 2.0 * (20.0 * a + 14.0 * (a ** 3))\n    else:\n        semitones = -(85.0 * a + 65.0 * (a ** 2))\n'''
if old not in s:
    raise SystemExit("expected monster YG mapping not found")
s = s.replace(old, new, 1)
old = '''    warp_gain = 1.30 + 0.55 * (a ** 2)\n'''
new = '''    if amount >= 0.0:\n        warp_gain = 1.55 + 0.95 * (a ** 2)\n    else:\n        warp_gain = 1.95 + 1.55 * a\n'''
if old not in s:
    raise SystemExit("expected monster YG warp gain not found")
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
PY
python3 -m py_compile "$TARGET"
pkill -f yuaz_ddsp_resampler.server 2>/dev/null || true
echo "Patched asymmetric YG test"
