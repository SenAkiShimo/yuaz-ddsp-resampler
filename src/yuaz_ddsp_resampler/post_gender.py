#!/usr/bin/env python3
import threading

import librosa
import numpy as np


_state = threading.local()


def set_gender_amount(value):
    _state.amount = float(np.clip(float(value), -100.0, 100.0))


def set_gender_f0(f0):
    if hasattr(f0, "detach"):
        f0 = f0.detach().cpu().numpy()
    _state.f0 = np.asarray(f0, dtype=np.float64).reshape(-1).copy()


def _take_context():
    amount = float(getattr(_state, "amount", 0.0))
    f0 = getattr(_state, "f0", None)
    _state.amount = 0.0
    _state.f0 = None
    return amount, f0


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


def _soft_endpoint_amount(amount):
    sign = np.sign(amount)
    value = abs(float(amount))
    if value <= 0.75:
        return sign * value
    x = (value - 0.75) / 0.25
    x = x * x * (3.0 - 2.0 * x)
    return sign * (0.75 + 0.15 * x)


def apply_final_gender_formant(audio, sr, target_f0, amount):
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    amount = float(np.clip(float(amount), -100.0, 100.0)) / 100.0
    if y.size < 256 or abs(amount) < 1e-7:
        return y.copy(), {"used": False, "amount": float(amount)}

    effective = _soft_endpoint_amount(amount)

    n_fft = 2048 if y.size >= 4096 else 1024
    hop = n_fft // 8
    spec = librosa.stft(
        y.astype(np.float64), n_fft=n_fft, hop_length=hop,
        win_length=n_fft, center=True,
    )
    mag = np.maximum(np.abs(spec), 1e-8)
    log_mag = np.log(mag)

    hz_per_bin = float(sr) / float(n_fft)
    smooth_bins = max(9, int(round(320.0 / max(hz_per_bin, 1e-6))))
    if smooth_bins % 2 == 0:
        smooth_bins += 1
    envelope = _smooth_frequency(log_mag, smooth_bins)

    curve = np.sign(effective) * (abs(effective) ** 0.85)
    semitones = 18.0 * curve

    freqs = librosa.fft_frequencies(sr=int(sr), n_fft=n_fft).astype(np.float64)
    low_guard = _smoothstep((freqs - 100.0) / 180.0)
    high_guard = 1.0 - _smoothstep((freqs - 6500.0) / 4200.0)
    local_semitones = semitones * low_guard * high_guard
    source_freq = freqs * np.power(2.0, local_semitones / 12.0)
    source_freq = np.clip(source_freq, 0.0, 0.5 * float(sr))

    warped = np.empty_like(envelope)
    for i in range(envelope.shape[1]):
        warped[:, i] = np.interp(
            source_freq, freqs, envelope[:, i],
            left=envelope[0, i], right=envelope[-1, i],
        )

    f0 = np.asarray(target_f0 if target_f0 is not None else [], dtype=np.float64).reshape(-1)
    if f0.size:
        src_x = np.linspace(0.0, 1.0, f0.size)
        dst_x = np.linspace(0.0, 1.0, envelope.shape[1])
        voiced = np.interp(dst_x, src_x, (f0 > 1.0).astype(np.float64))
        if voiced.size >= 7:
            voiced = np.convolve(voiced, np.ones(7, dtype=np.float64) / 7.0, mode="same")
        voiced = np.clip(voiced, 0.0, 1.0)
    else:
        voiced = np.ones(envelope.shape[1], dtype=np.float64)

    delta = 1.25 * (warped - envelope) * voiced[None, :]
    limit = 1.80
    delta = limit * np.tanh(delta / limit)
    ratio = np.exp(delta)
    shifted_spec = spec * ratio
    out = librosa.istft(
        shifted_spec, hop_length=hop, win_length=n_fft,
        length=y.size, center=True,
    ).astype(np.float32)

    rin = float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-12))
    rout = float(np.sqrt(np.mean(out.astype(np.float64) ** 2) + 1e-12))
    gain = 1.0
    if rin > 1e-8 and rout > 1e-8:
        gain = float(np.clip(rin / rout, 0.72, 1.38))
        out *= gain

    return out.astype(np.float32), {
        "used": True,
        "amount": float(amount),
        "effective_amount": float(effective),
        "semitones": float(semitones),
        "rms_gain": float(gain),
    }


def apply_context_gender(audio, sr):
    amount, target_f0 = _take_context()
    return apply_final_gender_formant(audio, sr, target_f0, amount)
