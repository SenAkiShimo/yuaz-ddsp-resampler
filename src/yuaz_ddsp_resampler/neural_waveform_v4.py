#!/usr/bin/env python3
"""Pitch-decoupled source-detail conditioning for conditioned-v4.

The v4 route deliberately does not expose raw voiced waveform samples to the
neural decoder.  Instead it extracts a coarse, frequency-smoothed source
spectral envelope plus temporal derivatives and broadband articulation cues.
Cross-pitch pair training presents these source cues together with a different
target F0, forcing the decoder to treat them as articulation/timbre evidence
rather than as a source-pitch carrier.
"""

import torch
import torch.nn.functional as F


SOURCE_DETAIL_N_FFT = 1024
SOURCE_DETAIL_HOP = 256
SOURCE_DETAIL_BANDS = 24
SOURCE_DETAIL_SMOOTH_BINS = 15
SOURCE_DETAIL_CHANNELS = SOURCE_DETAIL_BANDS * 2 + 3


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


def build_pitch_invariant_source_detail(
    source_waveform,
    frames,
    n_fft=SOURCE_DETAIL_N_FFT,
    hop=SOURCE_DETAIL_HOP,
    bands=SOURCE_DETAIL_BANDS,
    smooth_bins=SOURCE_DETAIL_SMOOTH_BINS,
):
    """Return coarse pitch-decoupled source detail as [B, C, frames].

    Features:
      * 24-band frequency-smoothed log spectral envelope (per-frame normalized)
      * temporal derivative of that envelope
      * broadband log-energy trajectory
      * broadband spectral-flux trajectory
      * spectral-flatness / aperiodicity proxy

    Frequency smoothing happens before band reduction so individual harmonic
    teeth are strongly attenuated.  No raw waveform or source F0 is returned.
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
    logmag = torch.log1p(mag)

    width = max(3, int(smooth_bins))
    if width % 2 == 0:
        width += 1
    smooth = F.avg_pool2d(
        logmag.unsqueeze(1),
        kernel_size=(width, 1),
        stride=1,
        padding=(width // 2, 0),
    ).squeeze(1)
    coarse = _resize_2d(smooth, bands, frames)
    mean = coarse.mean(dim=1, keepdim=True)
    std = coarse.std(dim=1, keepdim=True).clamp_min(1e-4)
    envelope = ((coarse - mean) / std).clamp(-4.0, 4.0)

    delta = torch.diff(envelope, dim=-1, prepend=envelope[..., :1])
    delta = delta.clamp(-2.5, 2.5)

    energy = torch.log(torch.sqrt(torch.mean(mag * mag, dim=1, keepdim=True) + 1e-8) + 1e-6)
    energy = _resize_time(energy, frames)
    energy = energy - energy.mean(dim=-1, keepdim=True)
    energy = energy / (energy.std(dim=-1, keepdim=True) + 1e-4)
    energy = energy.clamp(-4.0, 4.0)

    coarse_native = _resize_2d(smooth, bands, smooth.shape[-1])
    flux = torch.mean(
        torch.abs(torch.diff(coarse_native, dim=-1, prepend=coarse_native[..., :1])),
        dim=1,
        keepdim=True,
    )
    flux = _resize_time(flux, frames)
    flux = flux / (flux.mean(dim=-1, keepdim=True) + 1e-4)
    flux = flux.clamp(0.0, 6.0) / 3.0 - 1.0

    geometric = torch.exp(torch.mean(torch.log(mag + 1e-7), dim=1, keepdim=True))
    arithmetic = torch.mean(mag, dim=1, keepdim=True).clamp_min(1e-7)
    flatness = (geometric / arithmetic).clamp(0.0, 1.0)
    flatness = _resize_time(flatness, frames) * 2.0 - 1.0

    result = torch.cat([envelope, delta, energy, flux, flatness], dim=1)
    if result.shape[1] != int(SOURCE_DETAIL_CHANNELS):
        raise RuntimeError(
            f"source-detail width mismatch: expected {SOURCE_DETAIL_CHANNELS}, got {result.shape[1]}"
        )
    return result


def append_source_detail(base_conditioning, source_detail):
    if base_conditioning.ndim != 3 or source_detail.ndim != 3:
        raise ValueError("conditioning tensors must be [B, C, T]")
    if base_conditioning.shape[0] != source_detail.shape[0]:
        raise ValueError("conditioning batch sizes do not match")
    if source_detail.shape[-1] != base_conditioning.shape[-1]:
        source_detail = _resize_time(source_detail, base_conditioning.shape[-1])
    return torch.cat([base_conditioning, source_detail], dim=1)
