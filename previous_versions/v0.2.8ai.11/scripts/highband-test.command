#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -x "$ROOT/.venv/bin/python" ] || { echo "Run setup-macos.command first."; exit 1; }
strip_path() { python3 - "$1" <<'PY'
import shlex,sys
p=shlex.split(sys.argv[1].strip()); print(p[0] if p else '')
PY
}
echo "Yuaz 0.2.8ai.11 High-Band audit"
echo "Internal DDSP runs at 24 kHz (12 kHz Nyquist). YH is now restoration AMOUNT: 0..100."
echo "Drop one rendered WAV here (prefer YH100), then press Return; empty input runs only the algorithm self-check:"
read -r RAW || true
WAV="$(strip_path "${RAW:-}")"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$ROOT/.venv/bin/python" - "$WAV" <<'PY'
import sys, numpy as np
from pathlib import Path
from yuaz_ddsp_resampler.controls import parse_yuaz_controls
from yuaz_ddsp_resampler.learned_highband import synthesize_learned_highband

def band_ratio(x,sr,lo,hi):
    x=np.asarray(x,dtype=np.float64)
    n=min(len(x),262144)
    if n<512:return 0.0
    x=x[:n]*np.hanning(n)
    p=np.abs(np.fft.rfft(x))**2
    f=np.fft.rfftfreq(n,1/sr)
    total=float(p.sum())+1e-18
    m=(f>=lo)&(f<min(hi,sr/2))
    return float(np.sqrt(p[m].sum()/total)) if np.any(m) else 0.0

c=parse_yuaz_controls('YH100')
print(f'YH100 strength={c.highband_strength:.3f} auto_start={c.highband_yuaz_only_hz:.1f} Hz')
# Deliberately simulate a 24 kHz-DDSP-like body whose profile is dead above 12 kHz.
sr=44100;n=sr;t=np.arange(n)/sr;y=np.zeros(n,np.float32)
for k in range(1,60):
    f=220*k
    if f>=11500:break
    y += (0.08/(k**0.72)*np.sin(2*np.pi*f*t)).astype(np.float32)
prof={'band_centers_hz':[9000,11000,13000,15000,17000,19000],
      'voiced_db_to_full':[-28,-34,-80,-80,-80,-80],
      'unvoiced_db_to_full':[-30,-38,-80,-80,-80,-80],
      'voiced_harmonic_mix':0.72}
f0=np.full(200,220,np.float32)
o,st=synthesize_learned_highband(y,sr,f0,prof,123,assist_start_hz=c.highband_yuaz_only_hz,restoration_strength=c.highband_strength)
b=band_ratio(y,sr,13000,20000);a=band_ratio(o,sr,13000,20000)
print(f'Self-check 13-20 kHz ratio: before={b:.7f} after={a:.7f}')
print('Self-check stats:', {k:st.get(k) for k in ('restoration_strength','assist_start_hz','bridge_start_hz','reconstruction_floor_ratio','harmonic_count','branch_rms')})
if not (a>0.002 and a>b+0.001): raise SystemExit('FAIL: high-band restoration did not create measurable >13 kHz energy')
print('PASS: post-Nyquist restoration is active.')

wav=Path(sys.argv[1]).expanduser() if len(sys.argv)>1 and sys.argv[1] else None
if wav and wav.is_file():
    import soundfile as sf
    x,sr=sf.read(wav,always_2d=False)
    if np.ndim(x)>1:x=np.mean(x,axis=1)
    print('\nRendered WAV:',wav)
    print('sample rate:',sr,'Nyquist:',sr/2)
    for lo,hi in ((0,8000),(8000,12000),(12000,13000),(13000,16000),(16000,20000)):
        print(f'{lo/1000:>4.1f}-{hi/1000:>4.1f} kHz ratio = {band_ratio(x,sr,lo,hi):.7f}')
PY
