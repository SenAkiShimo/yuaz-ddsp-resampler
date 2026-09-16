#!/usr/bin/env python3
"""Fine high-band source conditioning for v0.4 HiFi."""

import torch
import torch.nn.functional as F


HIFI_N_FFT = 2048
HIFI_HOP = 256
HIFI_LOW_HZ = 6000.0
HIFI_HIGH_HZ = 22000.0
HIFI_BANDS = 64
HIFI_CHANNELS = HIFI_BANDS * 2 + 4


def _resize_2d(x, bands, frames):
    return F.interpolate(
        x.unsqueeze(1),
        size=(int(bands), int(frames)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def _resize_time(x, frames):
    if x.shape[-1] == int(frames):
        return x
    return F.interpolate(x, size=int(frames), mode="linear", align_corners=False)


def build_highband_source_detail(
    source_waveform,
    frames,
    sample_rate=48000,
    n_fft=HIFI_N_FFT,
    hop=HIFI_HOP,
    bands=HIFI_BANDS,
    low_hz=HIFI_LOW_HZ,
    high_hz=HIFI_HIGH_HZ,
):
    """Return mostly-unvoiced, high-resolution source texture as [B, C, T].

    Unlike the coarse v4 source envelope, this branch intentionally avoids
    frequency smoothing. A flatness/energy gate suppresses periodic harmonic
    detail so source-F0 comb structure is not passed through at full strength.
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
    bins = mag.shape[1]
    freqs = torch.linspace(
        0.0, float(sample_rate) * 0.5, bins,
        device=x.device, dtype=x.dtype,
    )
    high_mask = (freqs >= float(low_hz)) & (freqs <= float(high_hz))
    low_mask = (freqs >= 300.0) & (freqs < float(low_hz))
    if not bool(high_mask.any()):
        raise RuntimeError("high-band feature mask is empty")

    high_mag = mag[:, high_mask, :]
    high_log = torch.log1p(high_mag)

    # Preserve fine spectral shape. No frequency-domain averaging is applied.
    fine = _resize_2d(high_log, bands, frames)
    fine_mean = fine.mean(dim=1, keepdim=True)
    fine_std = fine.std(dim=1, keepdim=True).clamp_min(1e-4)
    texture = ((fine - fine_mean) / fine_std).clamp(-4.0, 4.0)

    geometric = torch.exp(torch.mean(torch.log(high_mag + 1e-7), dim=1, keepdim=True))
    arithmetic = torch.mean(high_mag, dim=1, keepdim=True).clamp_min(1e-7)
    flatness_native = (geometric / arithmetic).clamp(0.0, 1.0)

    high_rms_native = torch.sqrt(torch.mean(high_mag * high_mag, dim=1, keepdim=True) + 1e-8)
    if bool(low_mask.any()):
        low_mag = mag[:, low_mask, :]
        low_rms_native = torch.sqrt(torch.mean(low_mag * low_mag, dim=1, keepdim=True) + 1e-8)
    else:
        low_rms_native = torch.ones_like(high_rms_native)
    ratio_native = high_rms_native / (high_rms_native + low_rms_native + 1e-7)

    # Flat/noisy and high-band-rich frames receive the strongest fine texture.
    gate_native = torch.sigmoid(
        8.0 * (flatness_native - 0.18)
        + 10.0 * (ratio_native - 0.08)
        - 0.5
    )
    gate = _resize_time(gate_native, frames).clamp(0.0, 1.0)
    texture = texture * gate

    delta = torch.diff(texture, dim=-1, prepend=texture[..., :1]).clamp(-3.0, 3.0)

    high_energy = torch.log(high_rms_native + 1e-6)
    high_energy = _resize_time(high_energy, frames)
    high_energy = high_energy - high_energy.mean(dim=-1, keepdim=True)
    high_energy = high_energy / (high_energy.std(dim=-1, keepdim=True) + 1e-4)
    high_energy = high_energy.clamp(-4.0, 4.0)

    flux_native = torch.mean(
        torch.abs(torch.diff(high_log, dim=-1, prepend=high_log[..., :1])),
        dim=1,
        keepdim=True,
    )
    flux = _resize_time(flux_native, frames)
    flux = flux / (flux.mean(dim=-1, keepdim=True) + 1e-4)
    flux = flux.clamp(0.0, 6.0) / 3.0 - 1.0

    flatness = _resize_time(flatness_native, frames) * 2.0 - 1.0
    ratio = _resize_time(ratio_native, frames) * 2.0 - 1.0

    result = torch.cat(
        [texture, delta, high_energy, flux, flatness, ratio],
        dim=1,
    )
    if result.shape[1] != int(HIFI_CHANNELS):
        raise RuntimeError(
            f"HiFi detail width mismatch: expected {HIFI_CHANNELS}, got {result.shape[1]}"
        )
    return result


def append_highband_detail(base_conditioning, highband_detail):
    if base_conditioning.ndim != 3 or highband_detail.ndim != 3:
        raise ValueError("conditioning tensors must be [B, C, T]")
    if base_conditioning.shape[0] != highband_detail.shape[0]:
        raise ValueError("conditioning batch sizes do not match")
    if highband_detail.shape[-1] != base_conditioning.shape[-1]:
        highband_detail = _resize_time(highband_detail, base_conditioning.shape[-1])
    return torch.cat([base_conditioning, highband_detail], dim=1)
