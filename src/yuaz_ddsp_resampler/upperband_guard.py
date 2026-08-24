"""Upper-band continuity and output-rate terminal filtering.

The module provides slope-preserving crossover shaping, separate harmonic and
aperiodic terminal tapers, and a final output-rate guard. Earlier synthesis
paths remain available as runtime fallbacks.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from .post_gender import apply_context_gender


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _piecewise_profile(freqs: torch.Tensor, points) -> torch.Tensor:
    """Piecewise-linear profile, smoothed inside every interval."""
    out = torch.full_like(freqs, float(points[0][1]))
    for (f0, v0), (f1, v1) in zip(points[:-1], points[1:]):
        width = max(float(f1) - float(f0), 1.0)
        u = _smoothstep((freqs - float(f0)) / width)
        seg = float(v0) + (float(v1) - float(v0)) * u
        mask = (freqs >= float(f0)) & (freqs <= float(f1))
        out = torch.where(mask, seg, out)
        out = torch.where(freqs > float(f1), torch.full_like(out, float(v1)), out)
    return out


def terminal_frequencies(output_sample_rate: int) -> Dict[str, float]:
    """Return harmonic and aperiodic terminal frequencies for an output rate."""
    nyq = max(1000.0, 0.5 * float(output_sample_rate))
    harmonic_start = min(17600.0, nyq * 0.80)
    harmonic_zero = min(19800.0, nyq * 0.898)
    noise_start = min(18400.0, nyq * 0.835)
    noise_zero = min(21000.0, nyq * 0.953)
    harmonic_zero = min(harmonic_zero, nyq - 900.0)
    noise_zero = min(noise_zero, nyq - 600.0)
    harmonic_start = min(harmonic_start, harmonic_zero - 800.0)
    noise_start = min(noise_start, noise_zero - 700.0)
    return {
        "output_nyquist_hz": float(nyq),
        "harmonic_terminal_start_hz": float(harmonic_start),
        "harmonic_ceiling_hz": float(harmonic_zero),
        "noise_terminal_start_hz": float(noise_start),
        "terminal_zero_hz": float(noise_zero),
    }


def apply_upperband_spectral_guard(
    S_full: torch.Tensor,
    synthesis_sample_rate: int,
    output_sample_rate: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Apply a monotonic upper-band continuation profile and terminal taper."""
    if S_full.ndim != 3:
        raise ValueError("S_full must be [B, bins, frames]")
    bins = int(S_full.shape[1])
    device, dtype = S_full.device, S_full.dtype
    freqs = torch.linspace(
        0.0, 0.5 * float(synthesis_sample_rate), bins,
        device=device, dtype=dtype,
    ).view(1, bins, 1)
    tf = terminal_frequencies(output_sample_rate)

    profile = _piecewise_profile(freqs, [
        (0.0, 1.00),
        (11600.0, 1.00),
        (13200.0, 0.97),
        (15000.0, 0.90),
        (17000.0, 0.76),
        (18500.0, 0.56),
        (19800.0, 0.31),
        (tf["terminal_zero_hz"], 0.0),
        (0.5 * float(synthesis_sample_rate), 0.0),
    ])
    out = S_full * torch.clamp(profile, 0.0, 1.0)
    upper = freqs >= 12000.0
    mean_profile = float(profile[upper].detach().mean().cpu()) if bool(upper.any()) else 1.0
    return out, {
        "used": True,
        "mean_upper_spectral_guard": mean_profile,
        **tf,
    }


def apply_output_rate_mix_guard(
    harmonic_weight: torch.Tensor,
    noise_weight: torch.Tensor,
    synthesis_sample_rate: int,
    output_sample_rate: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Apply separate high-frequency tapers to harmonic and aperiodic weights."""
    if harmonic_weight.shape != noise_weight.shape or harmonic_weight.ndim != 3:
        raise ValueError("harmonic/noise weights must share [B, bins, frames]")
    bins = harmonic_weight.shape[1]
    device, dtype = harmonic_weight.device, harmonic_weight.dtype
    freqs = torch.linspace(
        0.0, 0.5 * float(synthesis_sample_rate), bins,
        device=device, dtype=dtype,
    ).view(1, bins, 1)
    tf = terminal_frequencies(output_sample_rate)

    presence = _piecewise_profile(freqs, [
        (0.0, 1.00),
        (10500.0, 1.00),
        (12500.0, 0.96),
        (14500.0, 0.88),
        (16500.0, 0.73),
        (18200.0, 0.54),
        (19800.0, 0.30),
        (tf["terminal_zero_hz"], 0.0),
        (0.5 * float(synthesis_sample_rate), 0.0),
    ])

    h_terminal = 1.0 - _smoothstep(
        (freqs - tf["harmonic_terminal_start_hz"]) /
        max(tf["harmonic_ceiling_hz"] - tf["harmonic_terminal_start_hz"], 1.0)
    )
    n_terminal = 1.0 - _smoothstep(
        (freqs - tf["noise_terminal_start_hz"]) /
        max(tf["terminal_zero_hz"] - tf["noise_terminal_start_hz"], 1.0)
    )
    h_gain = torch.clamp(presence * h_terminal, 0.0, 1.0)
    n_gain = torch.clamp(presence * n_terminal, 0.0, 1.0)
    h = harmonic_weight * h_gain
    n = noise_weight * n_gain

    top = freqs >= 20000.0
    denom = max(1, int(top.sum()))
    return h, n, {
        "used": True,
        "top_harmonic_gain_mean": float(h_gain[top].detach().sum().cpu()) / denom if bool(top.any()) else 0.0,
        "top_noise_gain_mean": float(n_gain[top].detach().sum().cpu()) / denom if bool(top.any()) else 0.0,
        **tf,
    }


def _final_gender(audio, sample_rate, stats):
    out, gender = apply_context_gender(audio, sample_rate)
    result = dict(stats)
    result["gender_used"] = bool(gender.get("used", False))
    result["gender_amount"] = float(gender.get("amount", 0.0))
    result["gender_effective_amount"] = float(gender.get("effective_amount", 0.0))
    result["gender_formant_semitones"] = float(gender.get("semitones", 0.0))
    return out, result


def apply_output_terminal_guard_numpy(
    audio,
    sample_rate: int,
    output_sample_rate: int | None = None,
):
    """Apply the final smooth terminal guard after high-band refinement."""
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    sr = int(sample_rate)
    out_sr = int(output_sample_rate or sample_rate)
    if y.size < 32:
        return _final_gender(y.copy(), sr, {"used": False, "reason": "too-short"})
    tf = terminal_frequencies(out_sr)
    nyq = 0.5 * float(sr)
    if nyq <= tf["noise_terminal_start_hz"] + 100.0:
        return _final_gender(y.copy(), sr, {"used": False, "reason": "insufficient-bandwidth", **tf})

    pad = min(max(512, int(round(0.040 * sr))), max(1, y.size - 1))
    yp = np.pad(y.astype(np.float64), (pad, pad), mode="reflect") if y.size > 1 else np.pad(y.astype(np.float64), (pad, pad))
    nfft = 1 << int(np.ceil(np.log2(max(64, yp.size))))
    spec = np.fft.rfft(yp, n=nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    u = np.clip(
        (freqs - tf["noise_terminal_start_hz"]) /
        max(tf["terminal_zero_hz"] - tf["noise_terminal_start_hz"], 1.0),
        0.0, 1.0,
    )
    smooth = u * u * (3.0 - 2.0 * u)
    gain = 1.0 - smooth
    gain[freqs >= tf["terminal_zero_hz"]] = 0.0
    out = np.fft.irfft(spec * gain, n=nfft)[:yp.size]
    out = out[pad:pad + y.size].astype(np.float32)
    return _final_gender(out, sr, {
        "used": True,
        "terminal_gain_at_20k": float(np.interp(20000.0, freqs, gain)),
        "terminal_gain_at_21k": float(np.interp(21000.0, freqs, gain)),
        **tf,
    })
