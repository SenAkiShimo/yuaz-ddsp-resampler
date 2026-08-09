import math

import numpy as np


def _resample_vector(x, n):
    x = np.asarray(x, dtype=np.float32)
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


def _moving_mean(x, width=3):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0 or width <= 1:
        return x.copy()
    width = int(max(1, width))
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def analyze_articulation_regions(audio, sr, hop, f0, detail, consonant_ms):
    audio = np.asarray(audio, dtype=np.float32)
    f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
    detail = np.asarray(detail, dtype=np.float32)
    hop_ms = float(hop) * 1000.0 / float(sr)
    total_ms = len(audio) * 1000.0 / float(sr)
    voiced = np.flatnonzero(f0 > 1.0)
    if voiced.size == 0:
        return {
            "first_voiced_frame": 0,
            "raw_end_frame": 0,
            "transition_end_frame": 0,
            "articulation_end_frame": 0,
            "first_voiced_ms": 0.0,
            "raw_end_ms": 0.0,
            "transition_end_ms": 0.0,
            "articulation_end_ms": 0.0,
            "fixed_region_ms": float(max(0.0, consonant_ms)),
            "confidence": 0.0,
        }

    first = int(voiced[0])
    last = int(voiced[-1])
    onset_ms = first * hop_ms
    raw_end_ms = max(0.0, onset_ms - 5.0)
    fixed_ms = float(np.clip(float(consonant_ms), 0.0, max(0.0, total_ms - hop_ms)))
    if fixed_ms <= 1.0:
        fixed_ms = min(total_ms * 0.35, onset_ms + 95.0)

    if detail.ndim == 2 and detail.shape[0] >= 1:
        flux = detail[-1].astype(np.float32)
    else:
        flux = np.zeros(max(1, len(f0)), dtype=np.float32)
    if len(flux) != len(f0):
        flux = _resample_vector(flux, len(f0))
    flux = _moving_mean(np.nan_to_num(flux, nan=0.0), 3)

    min_transition = first + max(2, int(round(35.0 / hop_ms)))
    max_transition = min(last, first + max(4, int(round(155.0 / hop_ms))))
    fixed_frame = int(round(fixed_ms / hop_ms))
    max_transition = min(last, max(max_transition, min(fixed_frame, first + int(round(190.0 / hop_ms)))))
    max_transition = max(min_transition, max_transition)

    post_a = min(len(flux) - 1, first + max(2, int(round(25.0 / hop_ms))))
    post_b = min(len(flux), first + max(5, int(round(210.0 / hop_ms))))
    reference = flux[post_a:post_b]
    if reference.size:
        threshold = float(np.percentile(reference, 42)) + 0.08
    else:
        threshold = float(np.median(flux)) + 0.08

    stable_frame = None
    run = max(2, int(round(22.0 / hop_ms)))
    for i in range(min_transition, max_transition + 1):
        j = min(len(flux), i + run)
        if j - i < run:
            break
        segment = flux[i:j]
        voiced_segment = f0[i:j] > 1.0
        if np.mean(voiced_segment) >= 0.75 and float(np.mean(segment)) <= threshold:
            stable_frame = i
            break
    if stable_frame is None:
        stable_frame = min(max_transition, max(min_transition, fixed_frame))

    transition_end_frame = int(np.clip(stable_frame, min_transition, last))
    transition_end_ms = transition_end_frame * hop_ms
    desired_end_ms = max(
        transition_end_ms + 35.0,
        min(fixed_ms + 22.0, onset_ms + 195.0),
        onset_ms + 92.0,
    )
    articulation_end_ms = min(desired_end_ms, onset_ms + 225.0, total_ms - hop_ms)
    articulation_end_ms = max(transition_end_ms + hop_ms, articulation_end_ms)
    articulation_end_frame = int(np.clip(round(articulation_end_ms / hop_ms), transition_end_frame + 1, last))
    articulation_end_ms = articulation_end_frame * hop_ms
    raw_end_frame = int(np.clip(round(raw_end_ms / hop_ms), 0, first))

    flux_span = float(np.mean(reference)) if reference.size else 0.0
    stable_flux = float(np.mean(flux[transition_end_frame:min(len(flux), transition_end_frame + run)])) if len(flux) else 0.0
    confidence = float(np.clip(0.45 + 0.35 * (stable_flux <= flux_span + 0.05) + 0.20 * (fixed_ms > onset_ms), 0.0, 1.0))
    return {
        "first_voiced_frame": first,
        "raw_end_frame": raw_end_frame,
        "transition_end_frame": transition_end_frame,
        "articulation_end_frame": articulation_end_frame,
        "first_voiced_ms": float(onset_ms),
        "raw_end_ms": float(raw_end_frame * hop_ms),
        "transition_end_ms": float(transition_end_ms),
        "articulation_end_ms": float(articulation_end_ms),
        "fixed_region_ms": float(fixed_ms),
        "confidence": confidence,
    }


def map_source_ms_to_target_ms(value_ms, source_fixed_ms, target_fixed_ms, source_total_ms, target_total_ms):
    value_ms = float(np.clip(value_ms, 0.0, max(0.0, source_total_ms)))
    source_fixed_ms = float(np.clip(source_fixed_ms, 0.0, max(0.0, source_total_ms)))
    target_fixed_ms = float(np.clip(target_fixed_ms, 0.0, max(0.0, target_total_ms)))
    if source_fixed_ms > 1e-5 and value_ms <= source_fixed_ms:
        return float(value_ms * target_fixed_ms / source_fixed_ms)
    source_tail = max(1e-5, source_total_ms - source_fixed_ms)
    target_tail = max(0.0, target_total_ms - target_fixed_ms)
    return float(target_fixed_ms + (value_ms - source_fixed_ms) * target_tail / source_tail)




def map_articulation_regions(regions, source_fixed_ms, target_fixed_ms, source_total_ms, target_total_ms):
    keys = {
        "raw_end_ms": "target_raw_end_ms",
        "first_voiced_ms": "target_onset_ms",
        "transition_end_ms": "target_transition_end_ms",
        "articulation_end_ms": "target_articulation_end_ms",
    }
    out = {}
    for source_key, target_key in keys.items():
        out[target_key] = map_source_ms_to_target_ms(
            float(regions.get(source_key, 0.0)),
            source_fixed_ms,
            target_fixed_ms,
            source_total_ms,
            target_total_ms,
        )
    out["target_raw_end_ms"] = min(out["target_raw_end_ms"], out["target_onset_ms"])
    out["target_transition_end_ms"] = max(out["target_onset_ms"], out["target_transition_end_ms"])
    out["target_articulation_end_ms"] = max(out["target_transition_end_ms"], out["target_articulation_end_ms"])
    return out


def _smooth_frequency(x, width=17):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 3:
        return x.copy()
    width = int(max(3, min(int(width), x.shape[0] if x.shape[0] % 2 else x.shape[0] - 1)))
    if width % 2 == 0:
        width -= 1
    if width <= 1:
        return x.copy()
    pad = width // 2
    padded = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    c = np.cumsum(padded, axis=0, dtype=np.float64)
    c = np.vstack([np.zeros((1, c.shape[1]), dtype=np.float64), c])
    return ((c[width:] - c[:-width]) / float(width)).astype(np.float32)


def _interp_time(matrix, frames):
    matrix = np.asarray(matrix, dtype=np.float32)
    frames = int(max(1, frames))
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-D")
    if matrix.shape[1] == frames:
        return matrix.copy()
    if matrix.shape[1] <= 1:
        return np.repeat(matrix[:, :1], frames, axis=1)
    src = np.linspace(0.0, 1.0, matrix.shape[1], dtype=np.float64)
    dst = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    out = np.empty((matrix.shape[0], frames), dtype=np.float32)
    for i in range(matrix.shape[0]):
        out[i] = np.interp(dst, src, matrix[i].astype(np.float64)).astype(np.float32)
    return out


def _raised_cosine_fade(n, fade_in=True):
    n = int(max(0, n))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    x = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float32)
    y = 0.5 - 0.5 * np.cos(np.pi * x)
    return y if fade_in else 1.0 - y


def _broad_frequency_trend(x, width=41):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 5:
        return np.zeros_like(x)
    width = int(max(5, min(int(width), x.shape[0] if x.shape[0] % 2 else x.shape[0] - 1)))
    if width % 2 == 0:
        width -= 1
    return _smooth_frequency(x, width)


def _neutralize_timbre(dynamic, sr):
    dynamic = np.asarray(dynamic, dtype=np.float32)
    if dynamic.ndim != 2 or dynamic.size == 0:
        return dynamic.copy()
    broad = _broad_frequency_trend(dynamic, 41 if dynamic.shape[0] >= 100 else 25)
    neutral = dynamic - 0.82 * broad
    neutral -= np.median(neutral, axis=0, keepdims=True)
    freqs = np.linspace(0.0, sr * 0.5, neutral.shape[0], dtype=np.float32)
    high = (freqs >= 3000.0) & (freqs <= min(9000.0, sr * 0.48))
    if np.any(high):
        high_mean = np.mean(neutral[high], axis=0)
        floor = -0.075
        lift = np.maximum(0.0, floor - high_mean).astype(np.float32)
        weight = np.clip((freqs - 2200.0) / 1300.0, 0.0, 1.0)
        if sr * 0.5 > 9000.0:
            weight *= np.clip((sr * 0.5 - freqs) / max(1.0, sr * 0.5 - 9000.0), 0.0, 1.0)
        neutral += weight[:, None] * lift[None, :]
    neutral = np.clip(neutral, -0.42, 0.42)
    return neutral.astype(np.float32)


def extract_neutral_articulation_template(source, sr, frames=32, n_fft=256):
    source = np.asarray(source, dtype=np.float32)
    if source.size < max(256, n_fft):
        return None
    import librosa
    hop = max(32, n_fft // 8)
    spec = librosa.stft(source, n_fft=n_fft, hop_length=hop, win_length=n_fft, window="hann", center=True)
    if spec.shape[1] < 3:
        return None
    logmag = np.log(np.abs(spec) + 1e-5).astype(np.float32)
    env = _smooth_frequency(logmag, 11)
    tail_frames = max(2, int(round(env.shape[1] * 0.28)))
    baseline = np.median(env[:, -tail_frames:], axis=1, keepdims=True).astype(np.float32)
    dynamic = env - baseline
    dynamic = _neutralize_timbre(dynamic, sr)
    energy = np.log(np.sqrt(np.mean(np.abs(spec) ** 2, axis=0)) + 1e-5).astype(np.float32)
    stable_energy = float(np.median(energy[-tail_frames:]))
    energy_delta = np.clip(energy - stable_energy, -0.24, 0.24)
    dynamic = _interp_time(dynamic, frames)
    energy_delta = _resample_vector(energy_delta, frames)
    return {
        "trajectory": dynamic.astype(np.float32),
        "energy_delta": energy_delta.astype(np.float32),
        "n_fft": int(n_fft),
        "frames": int(frames),
    }


def combine_canonical_articulation_templates(templates):
    valid = [t for t in templates if t is not None and np.asarray(t.get("trajectory")).ndim == 2]
    if not valid:
        return None
    shape = valid[0]["trajectory"].shape
    valid = [t for t in valid if t["trajectory"].shape == shape]
    if not valid:
        return None
    trajectories = np.stack([np.asarray(t["trajectory"], dtype=np.float32) for t in valid], axis=0)
    energies = np.stack([np.asarray(t["energy_delta"], dtype=np.float32) for t in valid], axis=0)
    canonical = np.median(trajectories, axis=0).astype(np.float32)
    energy = np.median(energies, axis=0).astype(np.float32)
    deviation = float(np.median(np.mean(np.abs(trajectories - canonical[None, ...]), axis=(1, 2)))) if len(valid) > 1 else 0.20
    coherence = float(np.clip(np.exp(-deviation / 0.16), 0.25, 1.0))
    return {
        "trajectory": canonical,
        "energy_delta": energy,
        "n_fft": int(valid[0].get("n_fft", 256)),
        "frames": int(canonical.shape[1]),
        "coherence": coherence,
        "source_count": int(len(valid)),
    }


def save_canonical_articulation(path, template, metadata=None):
    path = str(path)
    meta = dict(metadata or {})
    np.savez_compressed(
        path,
        trajectory=np.asarray(template["trajectory"], dtype=np.float16),
        energy_delta=np.asarray(template["energy_delta"], dtype=np.float16),
        n_fft=np.asarray(int(template.get("n_fft", 256))),
        frames=np.asarray(int(template.get("frames", np.asarray(template["trajectory"]).shape[1]))),
        coherence=np.asarray(float(template.get("coherence", 0.5)), dtype=np.float32),
        source_count=np.asarray(int(template.get("source_count", 1))),
        metadata=np.asarray(__import__("json").dumps(meta, ensure_ascii=False)),
    )


def load_canonical_articulation(path):
    with np.load(path, allow_pickle=False) as data:
        return {
            "trajectory": data["trajectory"].astype(np.float32),
            "energy_delta": data["energy_delta"].astype(np.float32),
            "n_fft": int(data["n_fft"].item()) if "n_fft" in data.files else 256,
            "frames": int(data["frames"].item()) if "frames" in data.files else int(data["trajectory"].shape[1]),
            "coherence": float(data["coherence"].item()) if "coherence" in data.files else 0.5,
            "source_count": int(data["source_count"].item()) if "source_count" in data.files else 1,
        }


def apply_articulation_template(template, target, sr, strength=0.80):
    target = np.asarray(target, dtype=np.float32)
    if template is None or target.size < 256:
        return target.copy(), False, {"trajectory_gain_rms_db": 0.0, "trajectory_strength": 0.0, "canonical_coherence": 0.0}
    import librosa
    n_fft = int(template.get("n_fft", 256))
    if target.size < n_fft:
        n_fft = 256 if target.size >= 256 else max(64, 2 ** int(np.floor(np.log2(target.size))))
    hop = max(32, n_fft // 8)
    tgt_spec = librosa.stft(target, n_fft=n_fft, hop_length=hop, win_length=n_fft, window="hann", center=True)
    if tgt_spec.shape[1] < 3:
        return target.copy(), False, {"trajectory_gain_rms_db": 0.0, "trajectory_strength": 0.0, "canonical_coherence": 0.0}
    dynamic = np.asarray(template["trajectory"], dtype=np.float32)
    if dynamic.shape[0] != tgt_spec.shape[0]:
        old = np.linspace(0.0, 1.0, dynamic.shape[0], dtype=np.float64)
        new = np.linspace(0.0, 1.0, tgt_spec.shape[0], dtype=np.float64)
        resized = np.empty((tgt_spec.shape[0], dynamic.shape[1]), dtype=np.float32)
        for i in range(dynamic.shape[1]):
            resized[:, i] = np.interp(new, old, dynamic[:, i].astype(np.float64)).astype(np.float32)
        dynamic = resized
    dynamic = _interp_time(dynamic, tgt_spec.shape[1])
    dynamic = _neutralize_timbre(dynamic, sr)
    energy_delta = _resample_vector(np.asarray(template.get("energy_delta", np.zeros(1)), dtype=np.float32), tgt_spec.shape[1])
    freqs = np.linspace(0.0, sr * 0.5, dynamic.shape[0], dtype=np.float32)
    band = np.ones_like(freqs)
    band[freqs < 120.0] = 0.0
    band[freqs > 8200.0] *= np.clip((sr * 0.5 - freqs[freqs > 8200.0]) / max(1.0, sr * 0.5 - 8200.0), 0.0, 1.0)
    coherence = float(np.clip(template.get("coherence", 0.55), 0.25, 1.0))
    effective_strength = float(strength) * (0.72 + 0.28 * coherence)
    time_strength = np.linspace(effective_strength, 0.30 * effective_strength, tgt_spec.shape[1], dtype=np.float32)
    gain_log = dynamic * band[:, None] * time_strength[None, :]
    gain_log += 0.34 * energy_delta[None, :] * time_strength[None, :]
    high = (freqs >= 3000.0) & (freqs <= min(9000.0, sr * 0.48))
    if np.any(high):
        high_mean = np.mean(gain_log[high], axis=0)
        floor = -0.055
        lift = np.maximum(0.0, floor - high_mean).astype(np.float32)
        weight = np.clip((freqs - 2400.0) / 1200.0, 0.0, 1.0)
        if sr * 0.5 > 9000.0:
            weight *= np.clip((sr * 0.5 - freqs) / max(1.0, sr * 0.5 - 9000.0), 0.0, 1.0)
        gain_log += weight[:, None] * lift[None, :]
    gain_log = np.clip(gain_log, -0.38, 0.38)
    shaped_spec = tgt_spec * np.exp(gain_log).astype(np.float32)
    shaped = librosa.istft(shaped_spec, hop_length=hop, win_length=n_fft, window="hann", center=True, length=len(target)).astype(np.float32)
    target_rms = float(np.sqrt(np.mean(target.astype(np.float64) ** 2) + 1e-12))
    shaped_rms = float(np.sqrt(np.mean(shaped.astype(np.float64) ** 2) + 1e-12))
    if target_rms > 1e-7 and shaped_rms > 1e-8:
        shaped *= float(np.clip(target_rms / shaped_rms, 10 ** (-1.0 / 20), 10 ** (1.0 / 20)))
    fade = min(len(target) // 4, int(round(0.010 * sr)))
    if fade > 2:
        w = np.ones(len(target), dtype=np.float32)
        w[:fade] = _raised_cosine_fade(fade, True)
        w[-fade:] = np.minimum(w[-fade:], _raised_cosine_fade(fade, False))
        shaped = target * (1.0 - w) + shaped * w
    gain_rms = float(np.sqrt(np.mean(gain_log.astype(np.float64) ** 2) + 1e-12))
    return shaped.astype(np.float32), True, {
        "trajectory_gain_rms_db": float(20.0 / np.log(10.0) * gain_rms),
        "trajectory_strength": float(effective_strength),
        "canonical_coherence": coherence,
    }


def transfer_articulation_trajectory(source, target, sr, strength=0.82):
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    template = extract_neutral_articulation_template(source, sr)
    if template is None:
        return target.copy(), False, {"trajectory_gain_rms_db": 0.0, "trajectory_strength": 0.0, "canonical_coherence": 0.0}
    template["coherence"] = 0.55
    template["source_count"] = 1
    return apply_articulation_template(template, target, sr, strength=strength)


def single_source_articulation_hybrid(original, generated, sr, regions, source_fixed_ms, target_fixed_ms, target_ms, canonical_template=None):
    original = np.asarray(original, dtype=np.float32)
    generated = np.asarray(generated, dtype=np.float32)
    n = len(generated)
    source_total_ms = len(original) * 1000.0 / float(sr)
    mapped = map_articulation_regions(regions, source_fixed_ms, target_fixed_ms, source_total_ms, target_ms)

    raw_src_ms = float(regions.get("raw_end_ms", max(0.0, regions.get("first_voiced_ms", 0.0) - 5.0)))
    onset_src_ms = float(regions.get("first_voiced_ms", 0.0))
    end_src_ms = float(regions.get("articulation_end_ms", onset_src_ms + 120.0))
    raw_src = int(np.clip(round(raw_src_ms * sr / 1000.0), 0, len(original)))
    onset_src = int(np.clip(round(onset_src_ms * sr / 1000.0), raw_src, len(original)))
    end_src = int(np.clip(round(end_src_ms * sr / 1000.0), onset_src, len(original)))

    raw_tgt = int(np.clip(round(mapped["target_raw_end_ms"] * sr / 1000.0), 0, n))
    onset_tgt = int(np.clip(round(mapped["target_onset_ms"] * sr / 1000.0), raw_tgt, n))
    end_tgt = int(np.clip(round(mapped["target_articulation_end_ms"] * sr / 1000.0), onset_tgt, n))

    out = generated.copy()
    trajectory_used = False
    source_kind = "none"
    trajectory_stats = {"trajectory_gain_rms_db": 0.0, "trajectory_strength": 0.0, "canonical_coherence": 0.0}
    if end_tgt - onset_tgt >= 256:
        confidence = float(np.clip(regions.get("confidence", 0.7), 0.0, 1.0))
        if canonical_template is not None:
            strength = 0.70 + 0.14 * confidence
            shaped, trajectory_used, trajectory_stats = apply_articulation_template(
                canonical_template, out[onset_tgt:end_tgt], sr, strength=strength
            )
            source_kind = "canonical"
        elif end_src - onset_src >= 256:
            strength = 0.58 + 0.16 * confidence
            shaped, trajectory_used, trajectory_stats = transfer_articulation_trajectory(
                original[onset_src:end_src], out[onset_tgt:end_tgt], sr, strength=strength
            )
            source_kind = "neutralized_local"
        else:
            shaped = out[onset_tgt:end_tgt]
        if trajectory_used:
            out[onset_tgt:end_tgt] = shaped

    if raw_src > 0 and raw_tgt > 0:
        raw = _resample_vector(original[:raw_src], raw_tgt)
        fade = min(int(round(0.012 * sr)), raw_tgt)
        out[:raw_tgt] = raw
        if fade > 2:
            a = raw_tgt - fade
            b = raw_tgt
            w = _raised_cosine_fade(fade, True)
            out[a:b] = raw[a:b] * (1.0 - w) + generated[a:b] * w

    return out.astype(np.float32), {
        "source_raw_end_ms": float(raw_src_ms),
        "source_onset_ms": float(onset_src_ms),
        "source_articulation_end_ms": float(end_src_ms),
        **mapped,
        "trajectory_transfer_used": bool(trajectory_used),
        "trajectory_source": source_kind,
        "single_periodic_source": True,
        "psola_used": False,
        "phase_shift_ms": 0.0,
        "hybrid_gain": 1.0,
        **trajectory_stats,
    }
