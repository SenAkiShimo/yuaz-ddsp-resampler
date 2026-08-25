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


def _install():
    global _installed, _original_articulation, _original_blend_v3
    if _installed:
        return
    from . import core

    _original_articulation = core.articulation_hybrid_mix
    _original_blend_v3 = core.blend_dualrate_fullband_body_v3

    def articulation_wrapper(original, generated, sr, source_f0, target_f0, regions, source_fixed_ms, target_fixed_ms, target_ms, canonical_template=None):
        mode = get_mode()
        if mode < 25.0:
            return _original_articulation(
                original, generated, sr, source_f0, target_f0, regions,
                source_fixed_ms, target_fixed_ms, target_ms,
                canonical_template=canonical_template,
            )

        baseline, baseline_stats = _original_articulation(
            original, generated, sr, source_f0, target_f0, regions,
            source_fixed_ms, target_fixed_ms, target_ms,
            canonical_template=canonical_template,
        )
        transition_ms = max(
            35.0,
            float(baseline_stats.get("target_articulation_end_ms", target_fixed_ms + 70.0)) - float(target_fixed_ms),
        )
        voiced = np.asarray(target_f0, dtype=np.float32)
        voiced = voiced[voiced > 1.0]
        target_f0_hz = float(np.median(voiced)) if voiced.size else 0.0
        mixed, phase_shift_ms, gain = core.hybrid_mix(
            original,
            generated,
            sr,
            float(target_fixed_ms),
            float(transition_ms),
            target_f0_hz=target_f0_hz,
        )
        stats = dict(baseline_stats)
        stats["phase_shift_ms"] = float(phase_shift_ms)
        stats["hybrid_gain"] = float(gain)
        stats["trajectory_transfer_used"] = False
        stats["trajectory_source"] = "clarity-ab-source-heavy"
        stats["clarity_ab_mode"] = float(mode)
        stats["clarity_ab_source_heavy"] = True
        return mixed.astype(np.float32), stats

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
