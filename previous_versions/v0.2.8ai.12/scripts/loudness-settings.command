#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -f config.json ]; then
  echo "Run scripts/configure-macos.command first."
  exit 1
fi
"$ROOT/.venv/bin/python" - <<'PY'
import json
from pathlib import Path
p=Path('config.json')
cfg=json.loads(p.read_text(encoding='utf-8'))
def ask(label, key, default):
    current=cfg.get(key, default)
    raw=input(f"{label} [{current}]: ").strip()
    return current if raw == '' else float(raw)
raw=input("Enable strict final-render loudness normalization? [Y/n] ").strip().lower()
cfg['normalize_voicebank_loudness']=raw not in ('n','no','0','false')
cfg['normalization_target_dbfs']=ask('Target active RMS dBFS', 'normalization_target_dbfs', -18.0)
cfg['normalization_peak_ceiling_dbfs']=ask('Peak ceiling dBFS', 'normalization_peak_ceiling_dbfs', -1.0)
cfg['normalization_peak_guard_knee_db']=abs(ask('Peak-guard knee width dB', 'normalization_peak_guard_knee_db', 3.0))
cfg['normalization_emergency_max_abs_gain_db']=abs(ask('Emergency maximum absolute gain dB', 'normalization_emergency_max_abs_gain_db', 30.0))
cfg['normalization_tolerance_db']=abs(ask('Target tolerance dB', 'normalization_tolerance_db', 0.05))
for key in ('normalization_max_gain_db','normalization_max_attenuation_db','normalization_render_trim_db'):
    cfg.pop(key, None)
p.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
print('Updated config.json. Re-run prepare-voicebank.command for each voicebank so the new runtime registry is written.')
PY
