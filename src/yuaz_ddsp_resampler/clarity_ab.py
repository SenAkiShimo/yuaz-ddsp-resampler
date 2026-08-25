#!/usr/bin/env python3
import threading

import numpy as np


_state = threading.local()
_original_articulation = None
_original_blend_v3 = None
_installed = False


def set_mode(value):
    global _installed
    _state.mode = float(np.clip(float(value), 0.0, 100.0))
    if not _installed:
        _install()


def get_mode():
    return float(getattr(_state, "mode", 0.0))


def _resample(x, n):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(max(0, n))
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if x.size == 0:
        return np.zeros(n, dtype=np.float32)
    if x.size == n:
        return x.copy()
    if x.size == 1:
        return np.full(n, float(x[0]), dtype=np.float32)
    src = np.linspace(0.0, 1.0, x.size, dtype=np.float64)
    dst = np.linspace(0.0, 1.0, n, dtype=np.float64)
    return np.interp(dst, src, x.astype(np.float64)).astype(np.float32)


def _local_source_f0(source_f0, source_samples, start_sample, end_sample):
    f0 = np.asarray(source_f0, dtype=np.float32).reshape(-1)
    if f0.size == 0 or source_samples <= 0:
        return 0.0
    a = int(np.clip(round(float(start_sample) / float(source_samples) * f0.size), 0, f0.size - 1))
    b = int(np.clip(round(float(end_sample) / float(source_samples) * f0.size), a + 1, f0.size))
    voiced = f0[a:b]
    voiced = voiced[voiced > 1.0]
    if voiced.size == 0:
        voiced = f0[f0 > 1.0]
    return float(np.median(voiced)) if voiced.size else 0.0


def _aperiodic_transient(source, sr, f0_hz):
    x = np.asarray(source, dtype=np.float32).reshape(-1)
    if x.size < 8:
        return np.zeros_like(x)
    residual = x.copy()
    if f0_hz > 35.0:
        delay = int(np.clip(round(float(sr) / float(f0_hz)), 2, max(2, x.size // 3)))
        if delay < x.size:
            residual[delay:] = x[delay:] - x[:-delay]
            residual[:delay] = 0.0
    width = max(3, int(round(0.00085 * float(sr))))
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width, dtype=np.float32) / float(width)
    smooth = np.convolve(residual, kernel, mode="same").astype(np.float32)
    residual = residual - smooth
    return residual.astype(np.float32)


def _inject_transient_detail(baseline, original, sr, source_f0, stats):
    out = np.asarray(baseline, dtype=np.float32).copy()
    source = np.asarray(original, dtype=np.float32).reshape(-1)
    if out.size < 16 or source.size < 16:
        return out, {"used": False}

    source_onset_ms = float(stats.get("source_onset_ms", 0.0))
    source_end_ms = float(stats.get("source_articulation_end_ms", source_onset_ms + 120.0))
    target_onset_ms = float(stats.get("target_onset_ms", 0.0))
    target_end_ms = float(stats.get("target_articulation_end_ms", target_onset_ms + 120.0))

    s0 = int(np.clip(round(source_onset_ms * sr / 1000.0), 0, source.size))
    s1 = int(np.clip(round(source_end_ms * sr / 1000.0), s0, source.size))
    t0 = int(np.clip(round(target_onset_ms * sr / 1000.0), 0, out.size))
    t1 = int(np.clip(round(target_end_ms * sr / 1000.0), t0, out.size))
    if s1 - s0 < 16 or t1 - t0 < 16:
        return out, {"used": False}

    local_f0 = _local_source_f0(source_f0, source.size, s0, s1)
    detail = _aperiodic_transient(source[s0:s1], sr, local_f0)
    detail = _resample(detail, t1 - t0)

    n = detail.size
    attack = min(n, max(1, int(round(0.010 * sr))))
    hold = min(n, max(attack, int(round(0.032 * sr))))
    envelope = np.ones(n, dtype=np.float32)
    if attack > 1:
        u = np.linspace(0.0, 1.0, attack, dtype=np.float32)
        envelope[:attack] = u * u * (3.0 - 2.0 * u)
    if n > hold:
        u = np.linspace(0.0, 1.0, n - hold, dtype=np.float32)
        envelope[hold:] = 0.5 + 0.5 * np.cos(np.pi * u)
    detail *= envelope

    base = out[t0:t1]
    base_rms = float(np.sqrt(np.mean(base.astype(np.float64) ** 2) + 1e-12))
    detail_rms = float(np.sqrt(np.mean(detail.astype(np.float64) ** 2) + 1e-12))
    if detail_rms <= 1e-8:
        return out, {"used": False}

    target_rms = max(0.004, min(0.32 * max(base_rms, 0.012), 0.055))
    gain = float(np.clip(target_rms / detail_rms, 0.0, 1.15))
    detail *= gain
    out[t0:t1] = np.clip(base.astype(np.float64) + detail.astype(np.float64), -1.2, 1.2).astype(np.float32)
    return out, {
        "used": True,
        "source_f0_hz": float(local_f0),
        "detail_rms": float(detail_rms),
        "gain": float(gain),
        "target_onset_ms": float(target_onset_ms),
        "target_end_ms": float(target_end_ms),
    }


def _install():
    global _installed, _original_articulation, _original_blend_v3
    if _installed:
        return
    from . import core

    _original_articulation = core.articulation_hybrid_mix
    _original_blend_v3 = core.blend_dualrate_fullband_body_v3

    def articulation_wrapper(original, generated, sr, source_f0, target_f0, regions, source_fixed_ms, target_fixed_ms, target_ms, canonical_template=None):
        mode = get_mode()
        baseline, stats = _original_articulation(
            original, generated, sr, source_f0, target_f0, regions,
            source_fixed_ms, target_fixed_ms, target_ms,
            canonical_template=canonical_template,
        )
        if mode < 25.0:
            return baseline, stats

        if 25.0 <= mode < 75.0:
            out = np.asarray(baseline, dtype=np.float32).copy()
            start = int(np.clip(round(float(stats.get("target_onset_ms", 0.0)) * sr / 1000.0), 0, out.size))
            end = int(np.clip(start + round(0.045 * sr), start, out.size))
            out[start:end] = 0.0
            result = dict(stats)
            result["clarity_ab_mode"] = float(mode)
            result["clarity_ab_sanity_cut"] = True
            return out, result

        mixed, detail_stats = _inject_transient_detail(
            baseline, original, sr, source_f0, stats
        )
        result = dict(stats)
        result["clarity_ab_mode"] = float(mode)
        result["clarity_ab_transient_used"] = bool(detail_stats.get("used", False))
        result["clarity_ab_source_f0_hz"] = float(detail_stats.get("source_f0_hz", 0.0))
        result["clarity_ab_transient_gain"] = float(detail_stats.get("gain", 0.0))
        return mixed.astype(np.float32), result

    def blend_v3_wrapper(legacy_output, fullband_output, sr, start_hz=8200.0, full_hz=13800.0):
        mode = get_mode()
        if mode < 75.0:
            return _original_blend_v3(
                legacy_output, fullband_output, sr,
                start_hz=start_hz, full_hz=full_hz,
            )
        y = np.asarray(legacy_output, dtype=np.float32).reshape(-1)
        return y.copy(), {
            "used": False,
            "backend": "clarity-ab-legacy-only",
            "reason": "clarity-ab-fullband-bypass",
            "crossover_start_hz": float(start_hz),
            "crossover_full_hz": float(full_hz),
            "fullband_branch_rms": 0.0,
            "fullband_safety_gain": 1.0,
        }

    core.articulation_hybrid_mix = articulation_wrapper
    core.blend_dualrate_fullband_body_v3 = blend_v3_wrapper
    _installed = True
