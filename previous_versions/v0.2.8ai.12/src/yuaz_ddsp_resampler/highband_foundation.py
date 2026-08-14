#!/usr/bin/env python3
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly


HIGHBAND_FOUNDATION_FORMAT = 1
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CROSSOVER_HZ = 9500.0
DEFAULT_FULL_HZ = 12100.0
DEFAULT_MAX_HZ = 22000.0


def _smoothstep(x):
    return x * x * (3.0 - 2.0 * x)


def _rms(x):
    return torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-8)


def _normalize_f0(f0, samples, device, dtype):
    if f0 is None:
        return torch.zeros((1, 1, samples), device=device, dtype=dtype)
    if not torch.is_tensor(f0):
        f0 = torch.as_tensor(f0, device=device, dtype=dtype)
    else:
        f0 = f0.to(device=device, dtype=dtype)
    if f0.ndim == 0:
        f0 = f0.view(1, 1, 1)
    elif f0.ndim == 1:
        f0 = f0.view(1, 1, -1)
    elif f0.ndim == 2:
        f0 = f0.unsqueeze(1)
    if f0.shape[-1] != samples:
        f0 = F.interpolate(f0, size=samples, mode="linear", align_corners=False)
    voiced = (f0 > 1.0).to(dtype)
    norm = torch.log2(torch.clamp(f0, min=45.0) / 220.0) / 3.0
    return torch.clamp(norm, -1.6, 1.6) * voiced


class HighBandFoundation(nn.Module):
    """Waveform residual network used only for bandwidth extension.

    Runtime code masks the prediction to the high band, so the foundation cannot
    rewrite the low-frequency Yuaz body. The loader remains compatible with the
    original v1 checkpoints; v2 training simply uses a wider receptive field.
    """

    def __init__(self, hidden=32, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.hidden = int(hidden)
        self.dilations = tuple(int(x) for x in dilations)
        self.in_proj = nn.Conv1d(2, self.hidden, 9, padding=4)
        self.blocks = nn.ModuleList([
            nn.Conv1d(self.hidden, self.hidden, 7, dilation=d, padding=3 * d)
            for d in self.dilations
        ])
        self.mix = nn.Conv1d(self.hidden, self.hidden, 1)
        self.out_proj = nn.Conv1d(self.hidden, 1, 9, padding=4)
        nn.init.kaiming_uniform_(self.in_proj.weight, a=0.2)
        nn.init.zeros_(self.in_proj.bias)
        for layer in self.blocks:
            nn.init.kaiming_uniform_(layer.weight, a=0.2)
            nn.init.zeros_(layer.bias)
        nn.init.kaiming_uniform_(self.mix.weight, a=0.2)
        nn.init.zeros_(self.mix.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, wave, f0=None):
        if wave.ndim == 2:
            wave = wave.unsqueeze(1)
        if wave.ndim != 3 or wave.shape[1] != 1:
            raise ValueError("wave must be [B, 1, T] or [B, T]")
        scale = _rms(wave).clamp(min=1e-4)
        x = torch.clamp(wave / scale, -4.0, 4.0)
        f = _normalize_f0(f0, x.shape[-1], x.device, x.dtype)
        if f.shape[0] == 1 and x.shape[0] > 1:
            f = f.expand(x.shape[0], -1, -1)
        h = F.silu(self.in_proj(torch.cat([x, f], dim=1)))
        for i, layer in enumerate(self.blocks):
            residual = F.silu(layer(h))
            h = h + (0.30 if i < 4 else 0.22) * residual
        h = F.silu(self.mix(h))
        residual = 0.45 * torch.tanh(self.out_proj(h)) * scale
        return residual


def spectral_band_mask(length, sr, crossover_hz=DEFAULT_CROSSOVER_HZ, full_hz=DEFAULT_FULL_HZ, max_hz=DEFAULT_MAX_HZ, device=None, dtype=torch.float32):
    freqs = torch.fft.rfftfreq(int(length), d=1.0 / float(sr), device=device)
    lo = torch.clamp((freqs - float(crossover_hz)) / max(1.0, float(full_hz) - float(crossover_hz)), 0.0, 1.0)
    rise = _smoothstep(lo)
    nyq = float(sr) * 0.5
    max_hz = min(float(max_hz), nyq - 100.0)
    fall_start = max(float(full_hz) + 800.0, max_hz - 1800.0)
    hi = torch.clamp((freqs - fall_start) / max(1.0, max_hz - fall_start), 0.0, 1.0)
    fall = 1.0 - _smoothstep(hi)
    mask = rise * fall
    mask = torch.where(freqs >= max_hz, torch.zeros_like(mask), mask)
    return mask.to(dtype=dtype)


def highpass_residual_torch(residual, sr, crossover_hz=DEFAULT_CROSSOVER_HZ, full_hz=DEFAULT_FULL_HZ, max_hz=DEFAULT_MAX_HZ):
    if residual.ndim == 2:
        residual = residual.unsqueeze(1)
    n = residual.shape[-1]
    spec = torch.fft.rfft(residual, dim=-1)
    mask = spectral_band_mask(n, sr, crossover_hz, full_hz, max_hz, residual.device, residual.dtype)
    out = torch.fft.irfft(spec * mask.view(1, 1, -1), n=n, dim=-1)
    return out


def _profile_shape_numpy(freqs, profile):
    if not profile:
        return np.ones_like(freqs, dtype=np.float64)
    try:
        centers = np.asarray(profile.get("band_centers_hz") or [], dtype=np.float64)
        db = np.asarray(profile.get("voiced_db_to_full") or [], dtype=np.float64)
        if centers.size < 2 or db.size != centers.size:
            return np.ones_like(freqs, dtype=np.float64)
        amp = np.power(10.0, db / 20.0)
        ref_mask = (centers >= 9000.0) & (centers <= 13000.0)
        ref = float(np.mean(amp[ref_mask])) if np.any(ref_mask) else float(np.mean(amp))
        shape = np.interp(freqs, centers, amp, left=amp[0], right=amp[-1]) / max(ref, 1e-7)
        return np.clip(shape, 0.55, 1.80)
    except Exception:
        return np.ones_like(freqs, dtype=np.float64)


def _resample_numpy(y, orig_sr, target_sr):
    y = np.asarray(y, dtype=np.float32)
    if int(orig_sr) == int(target_sr):
        return y
    g = math.gcd(int(orig_sr), int(target_sr))
    return resample_poly(y, int(target_sr)//g, int(orig_sr)//g).astype(np.float32)


def _soft_band_mask_numpy(freqs, rise_start, rise_full, fall_start, fall_end):
    freqs = np.asarray(freqs, dtype=np.float64)
    rise_x = np.clip((freqs - float(rise_start)) / max(1.0, float(rise_full) - float(rise_start)), 0.0, 1.0)
    rise = _smoothstep(rise_x)
    fall_x = np.clip((freqs - float(fall_start)) / max(1.0, float(fall_end) - float(fall_start)), 0.0, 1.0)
    fall = 1.0 - _smoothstep(fall_x)
    return rise * fall


def _filter_band_numpy(x, sr, rise_start, rise_full, fall_start, fall_end):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 8:
        return x.copy()
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sr))
    nyq = float(sr) * 0.5
    fall_end = min(float(fall_end), nyq - 40.0)
    fall_start = min(float(fall_start), fall_end - 80.0)
    mask = _soft_band_mask_numpy(freqs, rise_start, rise_full, fall_start, fall_end)
    return np.fft.irfft(spec * mask, n=x.size).real


def _moving_rms_numpy(x, window):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x.copy()
    window = max(3, min(int(window), x.size))
    left = window // 2
    right = window - 1 - left
    mode = "reflect" if x.size > 1 and min(left, right) < x.size else "edge"
    padded = np.pad(x * x, (left, right), mode=mode)
    cs = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    mean = (cs[window:] - cs[:-window]) / float(window)
    return np.sqrt(np.maximum(mean[:x.size], 0.0) + 1e-14)


def _coverage_against_reference(candidate, reference, sr):
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    if cand.size == 0 or ref.size != cand.size:
        return 0.0
    win = max(32, int(round(0.022 * float(sr))))
    cr = _moving_rms_numpy(cand, win)
    rr = _moving_rms_numpy(ref, win)
    active = rr > max(float(np.percentile(rr, 25)) * 0.35, 2e-7)
    if not np.any(active):
        return 1.0
    return float(np.mean(cr[active] >= np.maximum(rr[active] * 0.32, 2e-7)))


def _band_rms_numpy(x, sr, rise_start, rise_full, fall_start, fall_end):
    band = _filter_band_numpy(x, sr, rise_start, rise_full, fall_start, fall_end)
    return float(np.sqrt(np.mean(np.asarray(band, dtype=np.float64) ** 2) + 1e-12))


def _match_band_floor(component, reference_rms, sr, rise_start, rise_full, fall_start, fall_end, floor_ratio, max_gain_db=9.0):
    """Raise a band-limited component only when it falls below a conservative floor."""
    x = np.asarray(component, dtype=np.float64).reshape(-1)
    if x.size < 8 or reference_rms <= 1e-10:
        return x.copy(), 1.0, 0.0
    band = _filter_band_numpy(x, sr, rise_start, rise_full, fall_start, fall_end)
    band_rms = float(np.sqrt(np.mean(band ** 2) + 1e-12))
    target = float(reference_rms) * float(floor_ratio)
    if band_rms >= target or band_rms <= 1e-12:
        return x.copy(), 1.0, band_rms
    max_gain = 10.0 ** (float(max_gain_db) / 20.0)
    gain = float(np.clip(target / max(band_rms, 1e-12), 1.0, max_gain))
    # Only lift the requested band, not the whole component.
    return x + (gain - 1.0) * band, gain, band_rms


def blend_foundation_with_continuity(base, foundation_output, continuity_output, sr, strength=1.0):
    """True crossover between the 24 kHz Yuaz body and reconstructed upper band.

    Hotfix 1 still added high-band material on top of an untouched 24 kHz body.
    That can improve >12 kHz content but it cannot erase the body's visible
    Nyquist roof: the base spectrum itself still terminates abruptly. Hotfix 2
    uses a complementary seam crossover. It gently relaxes the 9.7–12.2 kHz
    body edge while the learned/neural branches take over the same overlap, then
    enforces conservative 12–15 and 15–18 kHz *band* floors derived from the
    actual 9–11 kHz edge energy. The floors scale existing learned/source texture
    rather than painting a broadband noise ceiling.
    """
    y = np.asarray(base, dtype=np.float32).reshape(-1)
    f = np.asarray(foundation_output, dtype=np.float32).reshape(-1)
    c = np.asarray(continuity_output, dtype=np.float32).reshape(-1)
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 1e-8 or y.size < 512 or f.size != y.size or c.size != y.size:
        return f.copy(), {"hybrid_used": False, "reason": "disabled-or-shape"}

    foundation_branch = f.astype(np.float64) - y.astype(np.float64)
    continuity_branch = c.astype(np.float64) - y.astype(np.float64)
    nyq = float(sr) * 0.5

    # Source-texture donor begins below the physical 12 kHz body ceiling so the
    # transition is an overlap, not a second disconnected layer.
    bridge = _filter_band_numpy(
        continuity_branch, sr,
        7900.0, 9000.0,
        min(13900.0, nyq - 850.0), min(15300.0, nyq - 160.0),
    )
    upper_ref = _filter_band_numpy(
        continuity_branch, sr,
        10800.0, 11900.0,
        min(20200.0, nyq - 1150.0), min(21900.0, nyq - 80.0),
    )
    foundation_upper = _filter_band_numpy(
        foundation_branch, sr,
        10600.0, 11800.0,
        min(20200.0, nyq - 1150.0), min(21900.0, nyq - 80.0),
    )

    win = max(64, int(round(0.020 * float(sr))))
    fr = _moving_rms_numpy(foundation_upper, win)
    rr = _moving_rms_numpy(upper_ref, win)
    ratio = fr / np.maximum(rr, 2e-7)
    gap = np.clip((1.02 - ratio) / 0.90, 0.0, 1.0)
    gap = _smoothstep(gap)

    bridge_weight = 0.94 * strength
    upper_weight = strength * (0.22 + 0.78 * gap)
    assist = bridge_weight * bridge + upper_weight * upper_ref
    combined_branch = foundation_branch + assist

    # The key seam fix: estimate the body edge before its 24 kHz Nyquist roof,
    # then make sure the reconstructed branch does not collapse immediately
    # above it. These are deliberately modest floors; a vowel should still get
    # darker toward 20 kHz rather than turning into a solid noise block.
    edge_rms = _band_rms_numpy(y, sr, 8200.0, 8800.0, 10500.0, 11200.0)
    combined_branch, seam_gain, seam_before = _match_band_floor(
        combined_branch, edge_rms, sr,
        10800.0, 11600.0, 13900.0, 15100.0,
        floor_ratio=0.30 * strength, max_gain_db=10.0,
    )
    combined_branch, air_gain, air_before = _match_band_floor(
        combined_branch, edge_rms, sr,
        13700.0, 15000.0, min(18100.0, nyq - 1800.0), min(19500.0, nyq - 500.0),
        floor_ratio=0.115 * strength, max_gain_db=8.0,
    )

    # Complementary body taper. The old additive design left the original
    # 24 kHz cutoff contour untouched, so a horizontal roof remained visible no
    # matter how much material existed above it. At YH100 this is only a few dB
    # at the very edge and fades in from ~9.7 kHz.
    body_edge_component = _filter_band_numpy(
        y, sr,
        9400.0, 10100.0,
        min(11950.0, nyq - 1100.0), min(12650.0, nyq - 450.0),
    )
    body_taper_amount = 0.26 * strength
    crossed_body = y.astype(np.float64) - body_taper_amount * body_edge_component

    body_rms = max(float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-12)), 1e-7)
    combined_rms = float(np.sqrt(np.mean(combined_branch ** 2) + 1e-12))
    safety_limit = body_rms * (0.23 + 0.13 * strength)
    safety_gain = 1.0
    if combined_rms > safety_limit:
        safety_gain = safety_limit / max(combined_rms, 1e-12)
        combined_branch *= safety_gain
        assist *= safety_gain
        combined_rms *= safety_gain

    before_cov = _coverage_against_reference(foundation_upper, upper_ref, sr)
    after_upper = _filter_band_numpy(
        combined_branch, sr,
        10800.0, 11900.0,
        min(20200.0, nyq - 1150.0), min(21900.0, nyq - 80.0),
    )
    after_cov = _coverage_against_reference(after_upper, upper_ref, sr)
    out = np.clip(crossed_body + combined_branch, -1.2, 1.2).astype(np.float32)
    return out, {
        "hybrid_used": True,
        "continuity_mode": "complementary-nyquist-crossover-v3",
        "continuity_assist_rms": float(np.sqrt(np.mean(assist ** 2) + 1e-12)),
        "continuity_bridge_weight": float(bridge_weight),
        "continuity_upper_weight_mean": float(np.mean(upper_weight)),
        "continuity_gap_gate_mean": float(np.mean(gap)),
        "upper_temporal_coverage_before": float(before_cov),
        "upper_temporal_coverage_after": float(after_cov),
        "combined_branch_rms": float(combined_rms),
        "hybrid_safety_gain": float(safety_gain),
        "nyquist_seam_edge_rms": float(edge_rms),
        "nyquist_seam_gain": float(seam_gain),
        "nyquist_air_gain": float(air_gain),
        "nyquist_seam_band_rms_before": float(seam_before),
        "nyquist_air_band_rms_before": float(air_before),
        "nyquist_body_taper_amount": float(body_taper_amount),
    }


def apply_highband_foundation(wave, sr, target_f0, model, profile=None, strength=1.0, crossover_hz=DEFAULT_CROSSOVER_HZ, full_hz=DEFAULT_FULL_HZ, max_hz=DEFAULT_MAX_HZ, device="cpu", conditioning_wave=None):
    y = np.asarray(wave, dtype=np.float32).reshape(-1)
    strength = float(np.clip(strength, 0.0, 1.0))
    if model is None or strength <= 1e-8 or y.size < 1024 or sr * 0.5 < 14000.0:
        return y.copy(), {"used": False, "reason": "disabled-or-unavailable"}
    model_sr = DEFAULT_SAMPLE_RATE
    cond = y if conditioning_wave is None else np.asarray(conditioning_wave, dtype=np.float32).reshape(-1)
    if cond.size < y.size:
        cond = np.pad(cond, (0, y.size - cond.size))
    cond = cond[:y.size]
    work = _resample_numpy(cond, sr, model_sr) if int(sr) != model_sr else cond.copy()
    dev = torch.device(device)
    wt = torch.from_numpy(work).to(dev).view(1, 1, -1)
    if target_f0 is None:
        f0 = None
    else:
        f0 = torch.as_tensor(np.asarray(target_f0, dtype=np.float32), device=dev).view(1, 1, -1)
    with torch.inference_mode():
        raw = model(wt, f0=f0)
        hb = highpass_residual_torch(raw, model_sr, crossover_hz, full_hz, max_hz)
    branch = hb[0, 0].detach().cpu().numpy().astype(np.float64)

    if profile:
        spec = np.fft.rfft(branch)
        freqs = np.fft.rfftfreq(branch.size, d=1.0 / float(model_sr))
        spec *= _profile_shape_numpy(freqs, profile)
        branch = np.fft.irfft(spec, n=branch.size).real

    body_rms = max(float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-12)), 1e-7)
    branch_rms = float(np.sqrt(np.mean(branch ** 2) + 1e-12))
    safety_limit = body_rms * (0.18 + 0.10 * strength)
    safety_gain = 1.0
    if branch_rms > safety_limit:
        safety_gain = safety_limit / max(branch_rms, 1e-12)
        branch *= safety_gain
        branch_rms *= safety_gain
    branch = (branch * strength).astype(np.float32)
    if int(sr) != model_sr:
        branch = _resample_numpy(branch, model_sr, sr)
        if branch.size < y.size:
            branch = np.pad(branch, (0, y.size-branch.size))
        branch = branch[:y.size]
    out = np.clip(y.astype(np.float64) + branch.astype(np.float64), -1.2, 1.2).astype(np.float32)
    return out, {
        "used": True,
        "backend": "highband-foundation-v1-compatible",
        "branch_rms": float(np.sqrt(np.mean(branch.astype(np.float64)**2)+1e-12)),
        "safety_gain": float(safety_gain),
        "model_sample_rate": int(model_sr),
        "output_sample_rate": int(sr),
        "crossover_hz": float(crossover_hz),
        "full_hz": float(full_hz),
        "max_hz": float(min(max_hz, model_sr * 0.5 - 100.0)),
        "conditioning_is_separate": bool(conditioning_wave is not None),
    }


def save_highband_foundation(path, model, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": HIGHBAND_FOUNDATION_FORMAT,
        "model": {"hidden": model.hidden, "dilations": list(model.dilations)},
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, path)
    return path


def load_highband_foundation(path, device="cpu"):
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if int(payload.get("format", 0)) != HIGHBAND_FOUNDATION_FORMAT:
        raise RuntimeError(f"Unsupported high-band foundation format: {payload.get('format')}")
    cfg = payload.get("model") or {}
    model = HighBandFoundation(hidden=int(cfg.get("hidden", 32)), dilations=tuple(cfg.get("dilations") or (1,2,4,8,16,32))).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    meta = dict(payload.get("metadata") or {})
    return model, meta


def inspect_highband_foundation(path):
    model, meta = load_highband_foundation(path, device="cpu")
    params = sum(p.numel() for p in model.parameters())
    return {"format": HIGHBAND_FOUNDATION_FORMAT, "parameters": int(params), "metadata": meta}
