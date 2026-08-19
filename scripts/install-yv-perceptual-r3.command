#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEV="$ROOT/src/yuaz_ddsp_resampler/vocal_controls.py"
DST_ROOT="$HOME/Library/Application Support/YuazDDSP/0.2.8ai.16"
RUNTIME="$DST_ROOT/src/yuaz_ddsp_resampler/vocal_controls.py"
RESAMPLERS="$HOME/Library/OpenUtau/Resamplers"
BASE_WRAPPER="$RESAMPLERS/Yuaz-DDSP-Resampler-v0.2.8ai.16-r2.sh"
BASE_MANIFEST="${BASE_WRAPPER%.sh}.yaml"
if [ ! -f "$BASE_WRAPPER" ]; then
  BASE_WRAPPER="$RESAMPLERS/Yuaz-DDSP-Resampler-v0.2.8ai.16.sh"
  BASE_MANIFEST="${BASE_WRAPPER%.sh}.yaml"
fi
R3_WRAPPER="$RESAMPLERS/Yuaz-DDSP-Resampler-v0.2.8ai.16-r3.sh"
R3_MANIFEST="${R3_WRAPPER%.sh}.yaml"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="$HOME/Documents/Yuaz-DDSP-Backups/ai16-yv-perceptual-r3-before-$STAMP"

for p in "$DEV" "$RUNTIME" "$BASE_WRAPPER" "$BASE_MANIFEST"; do
  [ -f "$p" ] || { echo "Required file missing: $p" >&2; exit 1; }
done
mkdir -p "$BACKUP" "$RESAMPLERS"
cp "$DEV" "$BACKUP/vocal_controls.dev.py"
cp "$RUNTIME" "$BACKUP/vocal_controls.runtime.py"
[ -f "$R3_WRAPPER" ] && cp "$R3_WRAPPER" "$BACKUP/$(basename "$R3_WRAPPER")"
[ -f "$R3_MANIFEST" ] && cp "$R3_MANIFEST" "$BACKUP/$(basename "$R3_MANIFEST")"

python3 - "$DEV" "$RUNTIME" <<'PY'
import sys
from pathlib import Path

paths=[Path(x) for x in sys.argv[1:]]
replacements=[
(
'''    voicing_pos_scale = carrier("voicing", 1.00)\n    voicing_neg_scale = carrier("voicing", 0.60)''',
'''    voicing_pos_scale = carrier("voicing", 1.25)\n    voicing_neg_scale = carrier("voicing", 0.70)'''
),
(
'''    voice_body = (\n        0.62 * torch.exp(-0.5 * torch.square((hz - 760.0) / 900.0))\n        + 0.38 * torch.exp(-0.5 * torch.square((hz - 2050.0) / 1550.0))\n    )\n    voicing_gain = torch.exp(\n        (0.72 * voicing_pos_eff - 0.28 * voicing_neg_eff) * voice_body * voiced\n    )''',
'''    voice_body = (\n        0.72 * torch.exp(-0.5 * torch.square((hz - 700.0) / 860.0))\n        + 0.48 * torch.exp(-0.5 * torch.square((hz - 1900.0) / 1350.0))\n    )\n    voice_presence = torch.exp(-0.5 * torch.square((hz - 3400.0) / 1750.0))\n    voice_air = torch.exp(-0.5 * torch.square((hz - 6500.0) / 2600.0))\n    voicing_shape = voice_body + 0.22 * voice_presence - 0.26 * voice_air\n    voicing_gain = torch.exp(\n        (0.95 * voicing_pos_eff - 0.38 * voicing_neg_eff) * voicing_shape * voiced\n    )'''
),
(
'''    v_pos_ap = torch.pow(torch.clamp(v_ap, 0.0, 1.0), 0.72) * voicing_pos_scale * voiced_ap\n    if float(torch.max(v_pos_ap).detach().cpu()) > 1e-7:\n        periodic_shape = 0.34 + 0.66 * torch.pow(ap_freq, 0.68)\n        out_ap = out_ap - 0.52 * v_pos_ap * periodic_shape * out_ap\n        out_ap = out_ap.clamp(0.012, 0.988)''',
'''    v_pos_ap = torch.pow(torch.clamp(v_ap, 0.0, 1.0), 0.68) * voicing_pos_scale * voiced_ap\n    v_neg_ap = torch.pow(torch.clamp(-v_ap, 0.0, 1.0), 0.72) * voicing_neg_scale * voiced_ap\n    if float(torch.max(v_pos_ap + v_neg_ap).detach().cpu()) > 1e-7:\n        periodic_shape = 0.30 + 0.70 * torch.pow(ap_freq, 0.66)\n        out_ap = out_ap - 0.72 * v_pos_ap * periodic_shape * out_ap\n        out_ap = out_ap + 0.44 * v_neg_ap * periodic_shape * (1.0 - out_ap)\n        out_ap = out_ap.clamp(0.012, 0.988)'''
),
(
'''    v_pos = torch.pow(torch.clamp(v_g, 0.0, 1.0), 0.72) * voiced_g * voicing_pos_scale\n    v_neg = torch.clamp(-v_g, 0.0, 1.0) * voiced_g * voicing_neg_scale\n    if float(torch.max(v_pos + v_neg).detach().cpu()) > 1e-7:\n        out_gate = out_gate + 0.92 * v_pos * (1.0 - out_gate)\n        out_gate = out_gate - 0.52 * v_neg * out_gate''',
'''    v_pos = torch.pow(torch.clamp(v_g, 0.0, 1.0), 0.68) * voiced_g * voicing_pos_scale\n    v_neg = torch.pow(torch.clamp(-v_g, 0.0, 1.0), 0.72) * voiced_g * voicing_neg_scale\n    if float(torch.max(v_pos + v_neg).detach().cpu()) > 1e-7:\n        out_gate = out_gate + 1.10 * v_pos * (1.0 - out_gate)\n        out_gate = out_gate - 0.58 * v_neg * out_gate'''
),
]

for path in paths:
    text=path.read_text(encoding='utf-8')
    for old,new in replacements:
        if old in text:
            if text.count(old)!=1:
                raise RuntimeError(f'ambiguous patch target in {path}: {old.splitlines()[0]}')
            text=text.replace(old,new,1)
        elif new in text:
            pass
        else:
            raise RuntimeError(f'expected YV block not found in {path}: {old.splitlines()[0]}')
    path.write_text(text,encoding='utf-8')
    compile(text,str(path),'exec')
    print('Patched:',path)
PY

cp "$BASE_WRAPPER" "$R3_WRAPPER"
cp "$BASE_MANIFEST" "$R3_MANIFEST"
chmod +x "$R3_WRAPPER"

PID="$(lsof -tiTCP:47888 -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$PID" ]; then
  kill "$PID" || true
  echo "Stopped ai16 runtime on port 47888: $PID"
fi

echo "Backup: $BACKUP"
echo "Installed fresh OpenUtau cache identity: $R3_WRAPPER"
grep -n 'voicing_pos_scale = carrier("voicing", 1.25)' "$RUNTIME"
grep -n 'voicing_shape = voice_body + 0.22 \* voice_presence - 0.26 \* voice_air' "$RUNTIME"
grep -n 'out_ap = out_ap - 0.72 \* v_pos_ap' "$RUNTIME"
grep -n 'out_gate = out_gate + 1.10 \* v_pos' "$RUNTIME"
echo "YV perceptual r3 installed. YT path and phonation model are unchanged."
echo "Restart OpenUtau, select Yuaz-DDSP-Resampler-v0.2.8ai.16-r3, then compare YV0 / YV100 / YV-100."
