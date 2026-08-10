import math


import numpy as np


DEFAULT_TARGET_DBFS = -18.0

DEFAULT_PEAK_CEILING_DBFS = -1.0

DEFAULT_PEAK_GUARD_KNEE_DB = 3.0

DEFAULT_EMERGENCY_MAX_ABS_GAIN_DB = 30.0

DEFAULT_TOLERANCE_DB = 0.05

DEFAULT_SILENCE_FLOOR_DBFS = -70.0


def db_to_gain(db):

    return float(10.0 ** (float(db) / 20.0))


def gain_to_db(gain):

    gain = max(float(gain), 1e-12)

    return float(20.0 * math.log10(gain))


def active_rms_dbfs(audio, sr, frame_ms=40.0, hop_ms=20.0):

    x = np.asarray(audio, dtype=np.float32).reshape(-1)

    if x.size == 0:

        return -120.0

    frame = max(64, int(round(float(sr) * float(frame_ms) / 1000.0)))

    hop = max(32, int(round(float(sr) * float(hop_ms) / 1000.0)))

    if x.size < frame:

        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))

        return max(-120.0, gain_to_db(rms))

    starts = np.arange(0, max(1, x.size - frame + 1), hop, dtype=np.int64)

    if starts.size == 0:

        starts = np.asarray([0], dtype=np.int64)

    if int(starts[-1]) + frame < x.size:

        starts = np.concatenate([starts, np.asarray([x.size - frame], dtype=np.int64)])

    frame_rms = np.empty(starts.size, dtype=np.float64)

    for i, start in enumerate(starts):

        seg = x[int(start):int(start) + frame].astype(np.float64)

        frame_rms[i] = np.sqrt(np.mean(seg * seg) + 1e-12)

    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-12))

    peak_frame_db = float(np.max(frame_db))

    gate_db = max(-55.0, peak_frame_db - 32.0)

    active = frame_db >= gate_db

    if not np.any(active):

        active[np.argmax(frame_db)] = True

    mask = np.zeros(x.size, dtype=bool)

    for use, start in zip(active, starts):

        if use:

            mask[int(start):min(x.size, int(start) + frame)] = True

    if not np.any(mask):

        mask[:] = True

    values = x[mask].astype(np.float64)

    rms = float(np.sqrt(np.mean(values * values) + 1e-12))

    return max(-120.0, gain_to_db(rms))


def oto_loudness_signature(offset, consonant, cutoff):

    return f"{float(offset):.3f}|{float(consonant):.3f}|{float(cutoff):.3f}"


def source_gain_to_target_db(measured_dbfs, target_dbfs=DEFAULT_TARGET_DBFS):

    return float(target_dbfs) - float(measured_dbfs)


def _soft_peak_guard(audio, peak_ceiling_dbfs=DEFAULT_PEAK_CEILING_DBFS, knee_db=DEFAULT_PEAK_GUARD_KNEE_DB):

    x = np.asarray(audio, dtype=np.float32)

    if x.size == 0:

        return x.copy(), 0

    ceiling = db_to_gain(float(peak_ceiling_dbfs))

    knee = ceiling * db_to_gain(-abs(float(knee_db)))

    span = max(ceiling - knee, 1e-6)

    mag = np.abs(x).astype(np.float64)

    mask = mag > knee

    if not np.any(mask):

        return x.copy(), 0

    guarded = mag.copy()

    guarded[mask] = knee + span * np.tanh((mag[mask] - knee) / span)

    y = np.copysign(guarded, x.astype(np.float64)).astype(np.float32)

    return y, int(np.count_nonzero(mask))


def normalize_final_render(

    audio,

    sr,

    target_dbfs=DEFAULT_TARGET_DBFS,

    peak_ceiling_dbfs=DEFAULT_PEAK_CEILING_DBFS,

    peak_guard_knee_db=DEFAULT_PEAK_GUARD_KNEE_DB,

    emergency_max_abs_gain_db=DEFAULT_EMERGENCY_MAX_ABS_GAIN_DB,

    tolerance_db=DEFAULT_TOLERANCE_DB,

    silence_floor_dbfs=DEFAULT_SILENCE_FLOOR_DBFS,

    max_iterations=6,

):

    x = np.asarray(audio, dtype=np.float32).copy()

    before = active_rms_dbfs(x, sr)

    peak_before = float(np.max(np.abs(x))) if x.size else 0.0

    if x.size == 0 or before <= float(silence_floor_dbfs):

        return x, {

            "used": False,

            "reason": "silence_or_empty",

            "target_dbfs": float(target_dbfs),

            "before_active_rms_dbfs": float(before),

            "after_active_rms_dbfs": float(before),

            "gain_db": 0.0,

            "target_error_db": float(target_dbfs) - float(before),

            "peak_before": peak_before,

            "peak_after": peak_before,

            "peak_guard_samples": 0,

            "iterations": 0,

            "target_reached": False,

            "safety_limited": False,

        }


    max_abs_gain = abs(float(emergency_max_abs_gain_db))

    total_gain_db = float(np.clip(float(target_dbfs) - before, -max_abs_gain, max_abs_gain))

    safety_limited = abs(total_gain_db - (float(target_dbfs) - before)) > 1e-8

    result = x

    guard_samples = 0

    after = before

    iterations = 0


    for iteration in range(max(1, int(max_iterations))):

        iterations = iteration + 1

        candidate = x * db_to_gain(total_gain_db)

        candidate, count = _soft_peak_guard(candidate, peak_ceiling_dbfs, peak_guard_knee_db)

        guard_samples = max(guard_samples, count)

        after = active_rms_dbfs(candidate, sr)

        error = float(target_dbfs) - after

        result = candidate

        if abs(error) <= float(tolerance_db):

            break

        proposed = total_gain_db + float(np.clip(error, -8.0, 8.0))

        clipped = float(np.clip(proposed, -max_abs_gain, max_abs_gain))

        if abs(clipped - proposed) > 1e-8:

            safety_limited = True

        if abs(clipped - total_gain_db) < 1e-7:

            break

        total_gain_db = clipped


    peak_after = float(np.max(np.abs(result))) if result.size else 0.0

    final_error = float(target_dbfs) - float(after)

    return result.astype(np.float32), {

        "used": True,

        "reason": "ok" if abs(final_error) <= float(tolerance_db) else "target_not_fully_reached",

        "target_dbfs": float(target_dbfs),

        "before_active_rms_dbfs": float(before),

        "after_active_rms_dbfs": float(after),

        "gain_db": float(total_gain_db),

        "target_error_db": float(final_error),

        "peak_before": peak_before,

        "peak_after": peak_after,

        "peak_ceiling_dbfs": float(peak_ceiling_dbfs),

        "peak_guard_samples": int(guard_samples),

        "iterations": int(iterations),

        "target_reached": bool(abs(final_error) <= float(tolerance_db)),

        "safety_limited": bool(safety_limited),

    }

