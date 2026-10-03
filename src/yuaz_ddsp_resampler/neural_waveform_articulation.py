#!/usr/bin/env python3
"""Pitch-robust temporal articulation conditioning for v0.4.1."""

import torch
import torch.nn.functional as F


ARTICULATION_N_FFT = 1024
ARTICULATION_HOP = 128
ARTICULATION_CHANNELS = 10
ARTICULATION_BANDS = (
    (300.0, 1000.0),
    (1000.0, 3000.0),
    (3000.0, 8000.0),
    (8000.0, 16000.0),
)


def _resize_time(x, frames):
    if x.shape[-1] == int(frames):
        return x
    return F.interpolate(x, size=int(frames), mode="linear", align_corners=False)


def _band_log_rms(mag, freqs, lo, hi):
    mask = (freqs >= float(lo)) & (freqs < float(hi))
    if not bool(mask.any()):
        return mag.new_zeros((mag.shape[0], 1, mag.shape[-1]))
    band = mag[:, mask, :]
    rms = torch.sqrt(torch.mean(band * band, dim=1, keepdim=True) + 1e-8)
    return torch.log(rms + 1e-6)


def _time_normalize(x, clamp=4.0):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(1e-4)
    return ((x - mean) / std).clamp(-float(clamp), float(clamp))


def build_articulation_contour(
    source_waveform,
    frames,
    sample_rate=48000,
    n_fft=ARTICULATION_N_FFT,
    hop=ARTICULATION_HOP,
):
    """Return source articulation contour [B, 10, T].

    The representation uses broad spectral-band energies and ratios rather than
    raw waveform/F0 detail. Its purpose is to preserve consonant duration,
    frication/burst timing, and consonant-to-vowel transition location.
    """
    x = source_waveform
    if x.ndim == 2:
        x = x.unsqueeze(1)
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError("source_waveform must be [B, 1, samples]")

    frames = max(1, int(frames))
    n_fft = int(n_fft)
    hop = int(hop)
    if x.shape[-1] < n_fft:
        x = F.pad(x, (0, n_fft - x.shape[-1]))

    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x[:, 0],
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = spec.abs().clamp_min(1e-7)
    freqs = torch.linspace(
        0.0, float(sample_rate) * 0.5, mag.shape[1],
        device=x.device, dtype=x.dtype,
    )

    broad = [_band_log_rms(mag, freqs, lo, hi) for lo, hi in ARTICULATION_BANDS]
    broad_norm = [_time_normalize(v) for v in broad]

    low = _band_log_rms(mag, freqs, 300.0, 3000.0)
    articulation = _band_log_rms(mag, freqs, 3000.0, 12000.0)
    high = _band_log_rms(mag, freqs, 8000.0, 16000.0)

    ratio_art = _time_normalize(articulation - low)
    ratio_high = _time_normalize(high - low)
    delta_art = torch.diff(ratio_art, dim=-1, prepend=ratio_art[..., :1]).clamp(-3.0, 3.0)
    delta_high = torch.diff(ratio_high, dim=-1, prepend=ratio_high[..., :1]).clamp(-3.0, 3.0)

    mask = (freqs >= 1000.0) & (freqs < 12000.0)
    mid = mag[:, mask, :] if bool(mask.any()) else mag
    geometric = torch.exp(torch.mean(torch.log(mid + 1e-7), dim=1, keepdim=True))
    arithmetic = torch.mean(mid, dim=1, keepdim=True).clamp_min(1e-7)
    flatness = (geometric / arithmetic).clamp(0.0, 1.0) * 2.0 - 1.0

    logmid = torch.log1p(mid)
    flux = torch.mean(
        torch.abs(torch.diff(logmid, dim=-1, prepend=logmid[..., :1])),
        dim=1,
        keepdim=True,
    )
    flux = flux / (flux.mean(dim=-1, keepdim=True) + 1e-4)
    flux = flux.clamp(0.0, 6.0) / 3.0 - 1.0

    channels = broad_norm + [ratio_art, ratio_high, delta_art, delta_high, flatness, flux]
    result = torch.cat([_resize_time(v, frames) for v in channels], dim=1)
    if result.shape[1] != ARTICULATION_CHANNELS:
        raise RuntimeError(
            f"articulation contour width mismatch: expected {ARTICULATION_CHANNELS}, got {result.shape[1]}"
        )
    return result


def append_articulation_contour(base_conditioning, contour):
    if base_conditioning.ndim != 3 or contour.ndim != 3:
        raise ValueError("conditioning tensors must be [B, C, T]")
    if base_conditioning.shape[0] != contour.shape[0]:
        raise ValueError("conditioning batch sizes do not match")
    if contour.shape[-1] != base_conditioning.shape[-1]:
        contour = _resize_time(contour, base_conditioning.shape[-1])
    return torch.cat([base_conditioning, contour], dim=1)
