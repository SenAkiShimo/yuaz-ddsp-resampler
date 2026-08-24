#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
./commands/run.command install-openutau-macos
RUNTIME="$HOME/Library/Application Support/YuazDDSP/0.2.9"
PKG="$RUNTIME/src/yuaz_ddsp_resampler"
CORE="$PKG/core.py"
VOCAL="$PKG/vocal_controls.py"
cat > "$PKG/post_gender.py" <<'PY'
import numpy as np
import librosa


def _smoothstep(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _smooth_frequency(x, width):
    width = max(3, int(width))
    if width % 2 == 0:
        width += 1
    pad = width // 2
    p = np.pad(np.asarray(x, dtype=np.float64), ((pad, pad), (0, 0)), mode="edge")
    c = np.cumsum(p, axis=0, dtype=np.float64)
    c = np.concatenate([np.zeros((1, c.shape[1]), dtype=np.float64), c], axis=0)
    return (c[width:] - c[:-width]) / float(width)


def apply_final_gender_formant(audio, sr, target_f0, amount):
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    amount = float(np.clip(float(amount), -100.0, 100.0)) / 100.0
    if y.size < 256 or abs(amount) < 1e-7:
        return y.copy(), {"used": False, "amount": float(amount)}

    n_fft = 2048 if y.size >= 4096 else 1024
    hop = n_fft // 8
    spec = librosa.stft(y.astype(np.float64), n_fft=n_fft, hop_length=hop, win_length=n_fft, center=True)
    mag = np.maximum(np.abs(spec), 1e-8)
    log_mag = np.log(mag)

    hz_per_bin = float(sr) / float(n_fft)
    smooth_bins = max(9, int(round(270.0 / max(hz_per_bin, 1e-6))))
    if smooth_bins % 2 == 0:
        smooth_bins += 1
    envelope = _smooth_frequency(log_mag, smooth_bins)

    a = abs(amount)
    semitones = np.sign(amount) * (20.0 * a + 14.0 * (a ** 3))

    freqs = librosa.fft_frequencies(sr=int(sr), n_fft=n_fft).astype(np.float64)
    low_guard = _smoothstep((freqs - 80.0) / 150.0)
    high_guard = 1.0 - _smoothstep((freqs - 7200.0) / 4500.0)
    local_semitones = semitones * low_guard * high_guard
    source_freq = freqs * np.power(2.0, local_semitones / 12.0)
    source_freq = np.clip(source_freq, 0.0, 0.5 * float(sr))

    warped = np.empty_like(envelope)
    for i in range(envelope.shape[1]):
        warped[:, i] = np.interp(source_freq, freqs, envelope[:, i], left=envelope[0, i], right=envelope[-1, i])

    f0 = np.asarray(target_f0, dtype=np.float64).reshape(-1)
    if f0.size:
        src_x = np.linspace(0.0, 1.0, f0.size)
        dst_x = np.linspace(0.0, 1.0, envelope.shape[1])
        voiced = np.interp(dst_x, src_x, (f0 > 1.0).astype(np.float64))
        if voiced.size >= 5:
            voiced = np.convolve(voiced, np.ones(5, dtype=np.float64) / 5.0, mode="same")
        voiced = np.clip(voiced, 0.0, 1.0)
    else:
        voiced = np.ones(envelope.shape[1], dtype=np.float64)

    warp_gain = 1.30 + 0.55 * (a ** 2)
    delta = warp_gain * (warped - envelope) * voiced[None, :]
    ratio = np.exp(np.clip(delta, -2.60, 2.60))
    shifted_spec = spec * ratio
    out = librosa.istft(shifted_spec, hop_length=hop, win_length=n_fft, length=y.size, center=True).astype(np.float32)

    rin = float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-12))
    rout = float(np.sqrt(np.mean(out.astype(np.float64) ** 2) + 1e-12))
    gain = 1.0
    if rin > 1e-8 and rout > 1e-8:
        gain = float(np.clip(rin / rout, 0.62, 1.62))
        out *= gain

    return out.astype(np.float32), {
        "used": True,
        "amount": float(amount),
        "semitones": float(semitones),
        "warp_gain": float(warp_gain),
        "rms_gain": float(gain),
    }
PY
python3 - "$CORE" "$VOCAL" <<'PY'
from pathlib import Path
import sys
core = Path(sys.argv[1])
vocal = Path(sys.argv[2])
s = core.read_text(encoding="utf-8")
old = 'from .loudness import normalize_final_render, oto_loudness_signature\n'
new = old + 'from .post_gender import apply_final_gender_formant\n'
if new not in s:
    if old not in s:
        raise SystemExit("core import patch point not found")
    s = s.replace(old, new, 1)
old = '''        gender_path = record.get("ai_gender_adapter")\n        if gender_path:\n            ai_paths.append(gender_path)\n'''
if old not in s:
    raise SystemExit("gender pack patch point not found")
s = s.replace(old, '        gender_path = None\n', 1)
old = '''            if self.ai13_upperband_guard_enabled and self.output_sr > 40000:\n                final, topband_guard_stats = apply_output_terminal_guard_numpy(\n                    final, self.output_sr, self.output_sr\n                )\n            loudness_stats = {\n'''
new = '''            if self.ai13_upperband_guard_enabled and self.output_sr > 40000:\n                final, topband_guard_stats = apply_output_terminal_guard_numpy(\n                    final, self.output_sr, self.output_sr\n                )\n            final, post_gender_stats = apply_final_gender_formant(\n                final, self.output_sr, target_f0, controls.gender_formant\n            )\n            loudness_stats = {\n'''
if old not in s:
    raise SystemExit("post-hybrid patch point not found")
s = s.replace(old, new, 1)
core.write_text(s, encoding="utf-8")

v = vocal.read_text(encoding="utf-8")
old = '    gender_scale = carrier("gender_formant", 0.85)\n'
if old not in v:
    raise SystemExit("internal gender patch point not found")
v = v.replace(old, '    gender_scale = 0.0\n', 1)
vocal.write_text(v, encoding="utf-8")
PY
python3 -m py_compile "$PKG/core.py" "$PKG/post_gender.py" "$PKG/vocal_controls.py"
pkill -f yuaz_ddsp_resampler.server 2>/dev/null || true
echo "Installed monster post-hybrid YG test"
