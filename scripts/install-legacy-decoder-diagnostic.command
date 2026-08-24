#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
./commands/run.command install-openutau-macos
RUNTIME="$HOME/Library/Application Support/YuazDDSP/0.2.9"
CORE="$RUNTIME/src/yuaz_ddsp_resampler/core.py"
python3 - "$CORE" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
needle = '''            decode_ms = (time.perf_counter() - decode_started) * 1000.0\n'''
block = '''            exact_out_samples = int(round(target_ms * self.output_sr / 1000.0))\n            diagnostic = resample_exact(generated, self.sr, self.output_sr, exact_out_samples)\n            write_wav(req["output"], diagnostic, self.output_sr, req.get("volume", 100))\n            return {\n                "ok": True,\n                "output": req["output"],\n                "source_sr": self.sr,\n                "output_sr": self.output_sr,\n                "target_ms": target_ms,\n                "diagnostic": "legacy-24k-decoder-only",\n                "yuaz_tension": float(controls.tension),\n            }\n'''
if needle not in s:
    raise SystemExit("diagnostic patch point not found")
s = s.replace(needle, block + needle, 1)
p.write_text(s, encoding="utf-8")
PY
pkill -f yuaz_ddsp_resampler.server 2>/dev/null || true
echo "Installed diagnostic: legacy 24 kHz decoder only"
