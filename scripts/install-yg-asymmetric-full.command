#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

./commands/run.command setup-macos

if [ ! -f config.json ]; then
  SRC="$HOME/Library/Application Support/YuazDDSP/0.2.9/config.json"
  if [ -f "$SRC" ]; then
    cp "$SRC" ./config.json
  else
    echo "config.json not found in worktree or installed 0.2.9 runtime." >&2
    exit 1
  fi
fi

./commands/run.command install-openutau-macos

RUNTIME="$HOME/Library/Application Support/YuazDDSP/0.2.9"
PKG="$RUNTIME/src/yuaz_ddsp_resampler"
CORE="$PKG/core.py"
VOCAL="$PKG/vocal_controls.py"
POST="$PKG/post_gender.py"

cat > "$POST" <<'PY'
import numpy as np
import librosa


def _smooth_frequency(x, width):
    width = max(3, int(width))
    if width % 2 == 0:
        width += 1
    pad = width // 2
    p = np.pad(np.asarray(x, dtype=np.float64), ((pad, pad), (0, 0)), mode="edge")
    c = np.cumsum(p, axis=0, dtype=np.float64)
    c = np.concatenate([np.zeros((1, c.shape[1]), dtype=np.float64), c], axis=0)
    return (c[width:] - c[:-width]) / float(width)


def _inverse_piecewise_frequency_map(freqs, nyq, scale, split_hz):
    scale = float(max(0.03, scale))
    split_hz = float(np.clip(split_hz, 180.0, nyq * 0.82))
    target_split = float(np.clip(split_hz * scale, 120.0, nyq * 0.86))
    source = np.empty_like(freqs, dtype=np.float64)
    low = freqs <= target_split
    source[low] = freqs[low] / scale
    denom = max(1e-6, nyq - target_split)
    source[~low] = split_hz + (freqs[~low] - target_split) * (nyq - split_hz) / denom
    source = np.clip(source, 0.0, nyq)
    guard = np.clip(freqs / 120.0, 0.0, 1.0)
    guard = guard * guard * (3.0 - 2.0 * guard)
    return freqs + (source - freqs) * guard


def apply_final_gender_formant(audio, sr, target_f0, amount):
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    raw = float(np.clip(float(amount), -100.0, 100.0))
    a = abs(raw) / 100.0
    if y.size < 256 or a < 1e-7:
        return y.copy(), {"used": False, "amount": raw}

    n_fft = 2048 if y.size >= 4096 else 1024
    hop = n_fft // 8
    spec = librosa.stft(y.astype(np.float64), n_fft=n_fft, hop_length=hop, win_length=n_fft, center=True)
    mag = np.maximum(np.abs(spec), 1e-8)
    log_mag = np.log(mag)

    hz_per_bin = float(sr) / float(n_fft)
    smooth_bins = max(9, int(round(250.0 / max(hz_per_bin, 1e-6))))
    if smooth_bins % 2 == 0:
        smooth_bins += 1
    envelope = _smooth_frequency(log_mag, smooth_bins)

    freqs = librosa.fft_frequencies(sr=int(sr), n_fft=n_fft).astype(np.float64)
    nyq = 0.5 * float(sr)

    if raw > 0.0:
        scale = 1.0 / (1.0 + 1.9 * a + 1.6 * a * a)
        split_hz = min(7200.0, nyq * 0.48)
        warp_gain = 1.45 + 1.05 * a * a
        route = "down"
    else:
        scale = 1.0 + 5.0 * a + 2.0 * a * a
        split_hz = min(4300.0, nyq * 0.70 / max(scale, 1.0))
        split_hz = max(1250.0, split_hz)
        warp_gain = 1.55 + 1.45 * a
        route = "up"

    source_freq = _inverse_piecewise_frequency_map(freqs, nyq, scale, split_hz)
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

    delta = warp_gain * (warped - envelope) * voiced[None, :]
    ratio = np.exp(np.clip(delta, -3.0, 3.0))
    shifted_spec = spec * ratio
    out = librosa.istft(shifted_spec, hop_length=hop, win_length=n_fft, length=y.size, center=True).astype(np.float32)

    rin = float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-12))
    rout = float(np.sqrt(np.mean(out.astype(np.float64) ** 2) + 1e-12))
    gain = 1.0
    if rin > 1e-8 and rout > 1e-8:
        gain = float(np.clip(rin / rout, 0.55, 1.80))
        out *= gain

    return out.astype(np.float32), {
        "used": True,
        "amount": raw,
        "route": route,
        "formant_scale": float(scale),
        "split_hz": float(split_hz),
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
imp = 'from .post_gender import apply_final_gender_formant\n'
anchor = 'from .loudness import normalize_final_render, oto_loudness_signature\n'
if imp not in s:
    if anchor not in s:
        raise SystemExit("core import patch point not found")
    s = s.replace(anchor, anchor + imp, 1)
old = '''        gender_path = record.get("ai_gender_adapter")\n        if gender_path:\n            ai_paths.append(gender_path)\n'''
if old in s:
    s = s.replace(old, '        gender_path = None\n', 1)
post = '''            final, post_gender_stats = apply_final_gender_formant(\n                final, self.output_sr, target_f0, controls.gender_formant\n            )\n'''
anchor2 = '''            loudness_stats = {\n'''
if post not in s:
    if anchor2 not in s:
        raise SystemExit("post-hybrid patch point not found")
    s = s.replace(anchor2, post + anchor2, 1)
core.write_text(s, encoding="utf-8")

v = vocal.read_text(encoding="utf-8")
old = '    gender_scale = carrier("gender_formant", 0.85)\n'
if old in v:
    v = v.replace(old, '    gender_scale = 0.0\n', 1)
vocal.write_text(v, encoding="utf-8")
PY

"$RUNTIME/.venv/bin/python" -m py_compile "$CORE" "$VOCAL" "$POST"
pkill -f yuaz_ddsp_resampler.server 2>/dev/null || true

test -f "$POST"
echo "Installed self-contained asymmetric YG test"
