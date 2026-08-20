#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -f config.json ] || { echo "Run scripts/configure-macos.command first."; exit 1; }

strip_path() {
  python3 - "$1" <<'PY'
import shlex,sys
parts=shlex.split(sys.argv[1].strip())
print(parts[0] if parts else "")
PY
}

WAV="${1:-}"
if [ -z "$WAV" ]; then
  echo "Drop one WAV from a prepared voicebank here, then press Return:"
  read -r RAW
  WAV="$(strip_path "$RAW")"
fi
[ -f "$WAV" ] || { echo "WAV not found: $WAV"; exit 1; }

TONE="${2:-G4}"
OUT="$ROOT/yf-yb-listen-test-output"
rm -rf "$OUT"
mkdir -p "$OUT"

render() {
  ./yuaz-ddsp-resampler "$WAV" "$OUT/$1.wav" "$TONE" 100 "$2" 0 3000 0 0 100 0 '!120' AA
}

render "00-baseline" ""
render "01-yb50" "YB50"
render "02-yb100" "YB100"
render "03-yf50" "YF50"
render "04-yf100" "YF100"
render "05-yb50-yf50" "YB50YF50"

"$ROOT/.venv/bin/python" - "$OUT" "$ROOT/logs/render_requests.jsonl" <<'PY'
import json,math,sys
from pathlib import Path
import numpy as np
import soundfile as sf

out=Path(sys.argv[1])
base,sr=sf.read(out/'00-baseline.wav',dtype='float32',always_2d=False)
base=np.asarray(base,dtype=np.float32).reshape(-1)
base_rms=float(np.sqrt(np.mean(base.astype(np.float64)**2)+1e-12))
print()
print('Waveform difference from baseline')
for p in sorted(out.glob('*.wav')):
    if p.name=='00-baseline.wav':
        continue
    x,xsr=sf.read(p,dtype='float32',always_2d=False)
    x=np.asarray(x,dtype=np.float32).reshape(-1)
    n=min(len(base),len(x))
    a=base[:n].astype(np.float64)
    b=x[:n].astype(np.float64)
    d=b-a
    dr=float(np.sqrt(np.mean(d*d)+1e-12))
    rel=20.0*math.log10(max(dr,1e-12)/max(base_rms,1e-12))
    den=float(np.linalg.norm(a)*np.linalg.norm(b))
    corr=float(np.dot(a,b)/den) if den>1e-12 else 0.0
    print(f'{p.stem:18s} diff_rms={dr:.8f}  diff_vs_base={rel:+.2f} dB  corr={corr:.8f}')

log=Path(sys.argv[2])
if log.is_file():
    rows=[]
    for line in log.read_text(encoding='utf-8',errors='ignore').splitlines():
        try:
            row=json.loads(line)
        except Exception:
            continue
        req=row.get('request') or {}
        if str(req.get('output','')).startswith(str(out)):
            rows.append(row)
    if rows:
        print()
        print('Render control trace')
        for row in rows[-6:]:
            req=row.get('request') or {}
            res=row.get('result') or {}
            effects=res.get('yuaz_ai_effects') or []
            technique={}
            route=''
            for effect in effects:
                names=effect.get('pack_controls') or []
                if 'falsetto' in names:
                    technique=effect.get('controls') or {}
                    route=str(effect.get('runtime_route') or '')
                    break
            name=Path(str(req.get('output',''))).stem
            print(f"{name:18s} flags={str(req.get('flags','')) or '(none)':10s} falsetto={float(technique.get('falsetto',0.0)):.3f} breathiness={float(technique.get('breathiness',0.0)):.3f} route={route or '-'}")
PY

echo
echo "Outputs: $OUT"
open "$OUT" 2>/dev/null || true
