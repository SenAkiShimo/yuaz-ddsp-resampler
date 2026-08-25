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


def _highpass(x, sr, cutoff_hz=1450.0):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size < 8:
        return np.zeros_like(x)
    width = int(round(0.443 * float(sr) / max(100.0, float(cutoff_hz))))
    width = max(5, width)
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width, dtype=np.float32) / float(width)
    low = np.convolve(x, kernel, mode="same").astype(np.float32)
    return (x - low).astype(np.float32)


def _aperiodic_transient(source, sr, f0_hz):
    x = np.asarray(source, dtype=np.float32).reshape(-1)
    if x.size < 8:
        return np.zeros_like(x)
    hp = _highpass(x, sr, 1450.0)
    if f0_hz <= 35.0:
        return hp
    period = int(np.clip(round(float(sr) / float(f0_hz)), 2, max(2, x.size // 3)))
    candidates = []
    for delta in (-1, 0, 1):
        delay = int(np.clip(period + delta, 2, max(2, x.size // 3)))
        residual = np.zeros_like(hp)
        if delay < hp.size:
            residual[delay:] = hp[delay:] - hp[:-delay]
        candidates.append(residual)
    stack = np.stack(candidates, axis=0)
    residual = np.median(stack, axis=0).astype(np.float32)
    derivative = np.zeros_like(hp)
    derivative[1:] = hp[1:] - hp[:-1]
    return (0.82 * residual + 0.18 * derivative).astype(np.float32)


def _inject_transient_detail(baseline, original, sr, source_f0, stats, strength=1.0):
    out = np.asarray(baseline, dtype=np.float32).copy()
    source = np.asarray(original, dtype=np.float32).reshape(-1)
    if out.size < 32 or source.size < 32:
        return out, {"used": False}

    source_raw_ms = float(stats.get("source_raw_end_ms", 0.0))
    source_onset_ms = float(stats.get("source_onset_ms", source_raw_ms + 5.0))
    source_end_ms = float(stats.get("source_articulation_end_ms", source_onset_ms + 120.0))
    target_raw_ms = float(stats.get("target_raw_end_ms", 0.0))
    target_onset_ms = float(stats.get("target_onset_ms", target_raw_ms + 5.0))
    target_end_ms = float(stats.get("target_articulation_end_ms", target_onset_ms + 120.0))

    source_end_ms = max(source_end_ms, source_onset_ms + 70.0)
    target_end_ms = max(target_end_ms, target_onset_ms + 70.0)
    source_start_ms = max(0.0, min(source_raw_ms, source_onset_ms - 4.0))
    target_start_ms = max(0.0, min(target_raw_ms, target_onset_ms - 4.0))

    s0 = int(np.clip(round(source_start_ms * sr / 1000.0), 0, source.size - 1))
    s1 = int(np.clip(round(source_end_ms * sr / 1000.0), s0 + 1, source.size))
    t0 = int(np.clip(round(target_start_ms * sr / 1000.0), 0, out.size - 1))
    t1 = int(np.clip(round(target_end_ms * sr / 1000.0), t0 + 1, out.size))

    if s1 - s0 < 32:
        s1 = min(source.size, s0 + max(32, int(round(0.090 * sr))))
    if t1 - t0 < 32:
        t1 = min(out.size, t0 + max(32, int(round(0.090 * sr))))
    if s1 - s0 < 32 or t1 - t0 < 32:
        return out, {"used": False}

    local_f0 = _local_source_f0(source_f0, source.size, s0, s1)
    detail = _aperiodic_transient(source[s0:s1], sr, local_f0)
    detail = _resample(detail, t1 - t0)

    n = detail.size
    attack = min(n, max(1, int(round(0.004 * sr))))
    onset = int(np.clip(round((target_onset_ms - target_start_ms) * sr / 1000.0), 0, n))
    decay_start = min(n, onset + max(1, int(round(0.055 * sr))))
    envelope = np.ones(n, dtype=np.float32)
    if attack > 1:
        envelope[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    if n > decay_start:
        u = np.linspace(0.0, 1.0, n - decay_start, dtype=np.float32)
        envelope[decay_start:] = 0.5 + 0.5 * np.cos(np.pi * u)
    detail *= envelope

    base = out[t0:t1]
    base_rms = float(np.sqrt(np.mean(base.astype(np.float64) ** 2) + 1e-12))
    detail_rms = float(np.sqrt(np.mean(detail.astype(np.float64) ** 2) + 1e-12))
    if detail_rms <= 1e-8:
        return out, {"used": False}

    amount = float(np.clip(strength, 0.0, 2.0))
    target_rms = max(0.010, min((0.24 + 0.10 * amount) * max(base_rms, 0.020), 0.060))
    gain = float(np.clip(target_rms / detail_rms, 0.0, 2.5)) * amount
    detail *= gain
    peak_cap = max(0.035, min(0.22, 0.72 * max(float(np.max(np.abs(base))), 0.05)))
    detail = np.clip(detail, -peak_cap, peak_cap)
    out[t0:t1] = np.clip(base.astype(np.float64) + detail.astype(np.float64), -1.2, 1.2).astype(np.float32)

    applied_rms = float(np.sqrt(np.mean(detail.astype(np.float64) ** 2) + 1e-12))
    return out, {
        "used": True,
        "source_f0_hz": float(local_f0),
        "detail_rms": float(detail_rms),
        "applied_rms": float(applied_rms),
        "gain": float(gain),
        "source_start_ms": float(source_start_ms),
        "target_start_ms": float(target_start_ms),
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

        strength = 0.78 if mode < 75.0 else 1.0
        mixed, detail_stats = _inject_transient_detail(
            baseline, original, sr, source_f0, stats, strength=strength
        )
        result = dict(stats)
        result["clarity_ab_mode"] = float(mode)
        result["clarity_ab_transient_used"] = bool(detail_stats.get("used", False))
        result["clarity_ab_source_f0_hz"] = float(detail_stats.get("source_f0_hz", 0.0))
        result["clarity_ab_transient_gain"] = float(detail_stats.get("gain", 0.0))
        result["clarity_ab_applied_rms"] = float(detail_stats.get("applied_rms", 0.0))
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
