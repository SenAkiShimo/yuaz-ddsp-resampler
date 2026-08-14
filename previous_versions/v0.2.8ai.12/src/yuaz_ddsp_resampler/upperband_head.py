"""0.2.8ai.12 frequency-dependent upper-band parameter head.

This module is intentionally additive.  The complete 0.2.8ai.11 source tree is
kept under previous_versions/v0.2.8ai.11 and the ai.11 full-band decoder remains
available as a runtime fallback.  ai.12 only supplies an alternate set of
48 kHz synthesis parameters for the upper band.

The frozen Yuaz checkpoint predicts acoustic parameters for a 24 kHz body.  A
48 kHz oscillator alone is therefore insufficient: if the old global
harmonic/noise gate is applied again after per-frequency aperiodicity, voiced
vowels can suppress almost the entire >12 kHz noise branch.  The parameter head
below keeps the trained low band untouched, extrapolates a conservative
frame-wise spectral envelope, raises aperiodicity gradually with frequency, and
turns the old scalar gate into a frequency-dependent context rather than a
second hard mute.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _band_mask(freqs: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return (freqs >= float(lo)) & (freqs <= float(hi))


def _band_log_mean(log_s: torch.Tensor, freqs: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    mask = _band_mask(freqs, lo, hi)
    if int(mask.sum()) < 1:
        return log_s[:, -1:, :].mean(dim=1, keepdim=True)
    return log_s[:, mask, :].mean(dim=1, keepdim=True)


def extend_spectral_envelope_upperband(
    S_lin: torch.Tensor,
    analysis_sample_rate: int,
    synthesis_sample_rate: int,
    synthesis_fft_size: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Extend the learned 24 kHz spectral envelope to the 48 kHz body.

    ai.11 used a single fixed decay curve.  ai.12 estimates the local upper-body
    spectral tilt for every frame and constrains it to a broad natural envelope.
    This retains voice-dependent brightness while preventing the extrapolated
    tail from collapsing to numerical zero immediately above 12 kHz.
    """

    if S_lin.ndim != 3:
        raise ValueError("S_lin must be [B, bins, frames]")
    B, C, T = S_lin.shape
    target_bins = int(synthesis_fft_size // 2 + 1)
    if target_bins <= C:
        return S_lin[:, :target_bins, :], {
            "used": False,
            "target_bins": int(target_bins),
            "source_bins": int(C),
        }

    device, dtype = S_lin.device, S_lin.dtype
    eps = 1e-7
    old_nyq = float(analysis_sample_rate) * 0.5
    new_nyq = float(synthesis_sample_rate) * 0.5
    old_freqs = torch.linspace(0.0, old_nyq, C, device=device, dtype=dtype)
    new_freqs = torch.linspace(0.0, new_nyq, target_bins, device=device, dtype=dtype)
    extra_freqs = new_freqs[C:].view(1, -1, 1)

    log_s = 20.0 * torch.log10(torch.clamp(S_lin, min=eps))
    low_db = _band_log_mean(log_s, old_freqs, 6500.0, 8100.0)
    high_db = _band_log_mean(log_s, old_freqs, 8800.0, 10600.0)
    anchor_db = _band_log_mean(log_s, old_freqs, 8200.0, 10800.0)

    low_center = 7300.0
    high_center = 9700.0
    octave_span = math.log2(high_center / low_center)
    slope_db_per_oct = (high_db - low_db) / max(octave_span, 1e-6)
    slope_db_per_oct = torch.clamp(slope_db_per_oct, -20.0, -1.5)

    anchor_hz = 9800.0
    rel_oct = torch.log2(torch.clamp(extra_freqs / anchor_hz, min=1.0))
    pred_rel_db = slope_db_per_oct * rel_oct

    # Broad guard rails: immediately above the old Nyquist, keep a useful but
    # restrained continuation; toward 20-22 kHz, allow the tail to become much
    # darker.  These are relative to the frame-wise 8.2-10.8 kHz edge.
    x = torch.clamp((extra_freqs - old_nyq) / max(new_nyq - old_nyq, 1.0), 0.0, 1.0)
    floor_rel_db = -3.5 - 23.0 * torch.pow(x, 1.22)
    ceiling_rel_db = -1.2 - 14.5 * torch.pow(x, 1.05)
    rel_db = torch.minimum(torch.maximum(pred_rel_db, floor_rel_db), ceiling_rel_db)

    # A little curvature keeps a vowel from becoming a flat ultrasonic shelf.
    rel_db = rel_db - 2.2 * torch.pow(x, 1.7)
    tail_db = anchor_db + rel_db
    tail = torch.pow(torch.tensor(10.0, device=device, dtype=dtype), tail_db / 20.0)

    # Anti-alias roll-off is deliberately delayed until ~21.5 kHz.  The target
    # OpenUtau output is normally 44.1 kHz, so the last few hundred Hz will be
    # discarded by the exact resampler anyway, but this keeps the 48 kHz body
    # well behaved when rendered directly.
    roll = 1.0 - _smoothstep((extra_freqs - 21400.0) / max(new_nyq - 21400.0, 1.0))
    tail = tail * torch.clamp(roll, min=0.0, max=1.0)

    out = torch.cat([S_lin, tail], dim=1)
    return out, {
        "used": True,
        "source_bins": int(C),
        "target_bins": int(target_bins),
        "analysis_nyquist_hz": float(old_nyq),
        "synthesis_nyquist_hz": float(new_nyq),
        "mean_slope_db_per_oct": float(slope_db_per_oct.detach().mean().cpu()),
    }


def extend_aperiodicity_upperband(
    A_lin: torch.Tensor,
    gate: torch.Tensor,
    synthesis_fft_size: int,
    analysis_sample_rate: int,
    synthesis_sample_rate: int,
    high_target: float = 0.82,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Extend AP without letting the old scalar gate erase the upper noise bed."""

    if A_lin.ndim != 3 or gate.ndim != 3:
        raise ValueError("A_lin and gate must be [B, bins/channels, frames]")
    B, C, T = A_lin.shape
    target_bins = int(synthesis_fft_size // 2 + 1)
    if target_bins <= C:
        return A_lin[:, :target_bins, :], {"used": False}

    device, dtype = A_lin.device, A_lin.dtype
    old_nyq = float(analysis_sample_rate) * 0.5
    new_nyq = float(synthesis_sample_rate) * 0.5
    extra = target_bins - C
    extra_freqs = torch.linspace(old_nyq, new_nyq, extra + 1, device=device, dtype=dtype)[1:].view(1, extra, 1)
    edge = A_lin[:, max(0, C - 32):C, :].mean(dim=1, keepdim=True)
    g = torch.clamp(gate, 0.0, 1.0)

    x = _smoothstep((extra_freqs - old_nyq) / max(new_nyq - old_nyq, 1.0))
    # Voiced frames still get a non-zero high-frequency aperiodic component;
    # unvoiced frames naturally approach an even higher AP target.
    target = float(high_target) + 0.10 * (1.0 - g)
    target = torch.clamp(target, 0.58, 0.96)
    start = torch.maximum(edge, 0.42 + 0.10 * (1.0 - g))
    tail = start + (target - start) * x
    tail = torch.clamp(tail, 0.02, 0.985)
    return torch.cat([A_lin, tail], dim=1), {
        "used": True,
        "mean_edge_ap": float(edge.detach().mean().cpu()),
        "mean_upper_ap": float(tail.detach().mean().cpu()),
    }


def frequency_dependent_mix_weights(
    A_spec: torch.Tensor,
    gate_spec: torch.Tensor,
    synthesis_sample_rate: int,
    synthesis_fft_size: int,
    head_start_hz: float = 8400.0,
    head_full_hz: float = 12400.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Convert Yuaz's scalar frame gate into frequency-dependent H/N weights.

    Below the crossover, ai.11 behaviour is retained.  Above it, the global gate
    becomes contextual rather than multiplicative: voiced frames keep harmonic
    support, but their noise branch receives a small non-zero floor so AP can
    actually create air/texture instead of being muted a second time.
    """

    if A_spec.ndim != 3 or gate_spec.ndim != 3:
        raise ValueError("A_spec and gate_spec must be [B, bins/channels, frames]")
    bins = A_spec.shape[1]
    device, dtype = A_spec.device, A_spec.dtype
    freqs = torch.linspace(0.0, 0.5 * float(synthesis_sample_rate), bins, device=device, dtype=dtype).view(1, bins, 1)
    upper = _smoothstep((freqs - float(head_start_hz)) / max(float(head_full_hz) - float(head_start_hz), 1.0))
    g = torch.clamp(gate_spec, 0.0, 1.0)

    low_h_context = g
    low_n_context = 1.0 - g
    high_h_context = 0.55 + 0.45 * g
    high_n_context = 0.30 + 0.70 * (1.0 - g)
    h_context = (1.0 - upper) * low_h_context + upper * high_h_context
    n_context = (1.0 - upper) * low_n_context + upper * high_n_context

    h = torch.clamp(1.0 - A_spec, 0.0, 1.0) * h_context
    n = torch.clamp(A_spec, 0.0, 1.0) * n_context

    # Prevent the pair from collapsing simply because both AP and the old gate
    # happen to attenuate the same frame.  At full upper-band takeover, the
    # desired combined magnitude stays around 0.68-0.80 depending on voicing.
    current = torch.sqrt(h * h + n * n + 1e-8)
    low_desired = current.detach()
    upper_desired = 0.68 + 0.12 * (1.0 - g)
    desired = (1.0 - upper) * low_desired + upper * upper_desired
    scale = torch.clamp(desired / torch.clamp(current, min=1e-4), 0.72, 2.8)
    h = h * scale
    n = n * scale

    upper_mask = (freqs >= float(head_full_hz)).to(dtype)
    denom = torch.clamp(upper_mask.sum() * A_spec.shape[0] * A_spec.shape[2], min=1.0)
    return h, n, {
        "used": True,
        "head_start_hz": float(head_start_hz),
        "head_full_hz": float(head_full_hz),
        "upper_harmonic_weight_mean": float((h * upper_mask).detach().sum().cpu() / denom.cpu()),
        "upper_noise_weight_mean": float((n * upper_mask).detach().sum().cpu() / denom.cpu()),
        "upper_mix_scale_mean": float((scale * upper_mask).detach().sum().cpu() / denom.cpu()),
    }
