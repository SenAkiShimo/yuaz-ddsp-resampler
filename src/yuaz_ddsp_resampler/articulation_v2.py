import numpy as np

from .articulation import (
    _interp_time,
    _raised_cosine_fade,
    _resample_vector,
    extract_neutral_articulation_template,
    map_articulation_regions,
)


def _resize_frequency(dynamic, bins):
    dynamic = np.asarray(dynamic, dtype=np.float32)
    if dynamic.shape[0] == int(bins):
        return dynamic.copy()
    old = np.linspace(0.0, 1.0, dynamic.shape[0], dtype=np.float64)
    new = np.linspace(0.0, 1.0, int(bins), dtype=np.float64)
    out = np.empty((int(bins), dynamic.shape[1]), dtype=np.float32)
    for i in range(dynamic.shape[1]):
        out[:, i] = np.interp(new, old, dynamic[:, i].astype(np.float64)).astype(np.float32)
    return out


def _trajectory_contrast(dynamic):
    dynamic = np.asarray(dynamic, dtype=np.float32)
    if dynamic.ndim != 2 or dynamic.size == 0:
        return 0.0
    early = max(2, int(round(dynamic.shape[1] * 0.38)))
    values = np.abs(dynamic[:, :early]).reshape(-1)
    return float(np.percentile(values, 72)) if values.size else 0.0


def _transition_strength_curve(frames, effective_strength, transition_fraction):
    frames = int(max(1, frames))
    boundary = int(np.clip(round(float(transition_fraction) * max(1, frames - 1)), 1, max(1, frames - 1)))
    curve = np.empty(frames, dtype=np.float32)
    if boundary > 0:
        curve[:boundary] = np.linspace(1.14, 1.00, boundary, dtype=np.float32)
    remain = frames - boundary
    if remain > 0:
        x = np.linspace(0.0, 1.0, remain, dtype=np.float32)
        smooth = x * x * (3.0 - 2.0 * x)
        curve[boundary:] = 1.00 - 0.80 * smooth
    return curve * float(effective_strength)


def apply_articulation_template_v2(
    template,
    target,
    sr,
    strength=0.80,
    transition_fraction=0.35,
    voiced_onset_zero=False,
):
    """Apply an already-neutralized articulation trajectory without neutralizing it twice.

    The v1 runtime passed canonical trajectories through _neutralize_timbre a second
    time. That is safe for timbre but can erase broad voiced-consonant cues such as
    nasal/formant transitions. v2 keeps the stored pitch-independent trajectory,
    re-centers only its stable tail, and concentrates the strongest transfer before
    the detected consonant-to-vowel transition. It never injects raw voiced waveform,
    so source F0 cannot leak into the generated body.
    """
    target = np.asarray(target, dtype=np.float32)
    empty = {
        "trajectory_gain_rms_db": 0.0,
        "trajectory_strength": 0.0,
        "canonical_coherence": 0.0,
        "trajectory_contrast": 0.0,
        "trajectory_revision": 2,
        "voiced_onset_zero": bool(voiced_onset_zero),
    }
    if template is None or target.size < 256:
        return target.copy(), False, empty

    import librosa

    n_fft = int(template.get("n_fft", 256))
    if target.size < n_fft:
        n_fft = 256 if target.size >= 256 else max(64, 2 ** int(np.floor(np.log2(target.size))))
    hop = max(32, n_fft // 8)
    tgt_spec = librosa.stft(
        target,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window="hann",
        center=True,
    )
    if tgt_spec.shape[1] < 3:
        return target.copy(), False, empty

    dynamic = np.asarray(template["trajectory"], dtype=np.float32)
    dynamic = _resize_frequency(dynamic, tgt_spec.shape[0])
    dynamic = _interp_time(dynamic, tgt_spec.shape[1])

    # Canonical templates were already neutralized when prepared. Remove only a
    # residual stable-tail offset here; do not suppress the broad trajectory again.
    tail = max(2, int(round(dynamic.shape[1] * 0.20)))
    dynamic = dynamic - np.median(dynamic[:, -tail:], axis=1, keepdims=True).astype(np.float32)
    dynamic = np.clip(dynamic, -0.48, 0.48)

    contrast = _trajectory_contrast(dynamic)
    contrast_norm = float(np.clip((contrast - 0.035) / 0.16, 0.0, 1.0))
    coherence = float(np.clip(template.get("coherence", 0.55), 0.25, 1.0))
    effective_strength = float(strength) * (0.78 + 0.22 * coherence) * (1.0 + 0.18 * contrast_norm)
    if bool(voiced_onset_zero) and contrast_norm > 0.18:
        effective_strength *= 1.08
    effective_strength = float(np.clip(effective_strength, 0.0, 1.18))

    time_strength = _transition_strength_curve(
        tgt_spec.shape[1],
        effective_strength,
        float(np.clip(transition_fraction, 0.10, 0.78)),
    )
    energy_delta = _resample_vector(
        np.asarray(template.get("energy_delta", np.zeros(1)), dtype=np.float32),
        tgt_spec.shape[1],
    )

    freqs = np.linspace(0.0, sr * 0.5, dynamic.shape[0], dtype=np.float32)
    band = np.ones_like(freqs)
    band[freqs < 100.0] = 0.0
    above = freqs > 8500.0
    if np.any(above):
        band[above] *= np.clip(
            (sr * 0.5 - freqs[above]) / max(1.0, sr * 0.5 - 8500.0),
            0.0,
            1.0,
        )

    # Slightly favor the low/mid trajectory for strong voiced-onset consonants.
    low_mid = np.ones_like(freqs)
    if bool(voiced_onset_zero) and contrast_norm > 0.18:
        low_mid *= 1.0 + 0.12 * np.clip((3600.0 - freqs) / 3000.0, 0.0, 1.0)

    gain_log = dynamic * band[:, None] * low_mid[:, None] * time_strength[None, :]
    gain_log += 0.46 * energy_delta[None, :] * time_strength[None, :]

    # Do not let articulation transfer create a new high-frequency wall.
    high = freqs >= 7000.0
    if np.any(high):
        gain_log[high] = np.clip(gain_log[high], -0.42, 0.30)
    limit = 0.56 if bool(voiced_onset_zero) and contrast_norm > 0.18 else 0.50
    gain_log = np.clip(gain_log, -limit, limit)

    shaped_spec = tgt_spec * np.exp(gain_log).astype(np.float32)
    shaped = librosa.istft(
        shaped_spec,
        hop_length=hop,
        win_length=n_fft,
        window="hann",
        center=True,
        length=len(target),
    ).astype(np.float32)

    target_rms = float(np.sqrt(np.mean(target.astype(np.float64) ** 2) + 1e-12))
    shaped_rms = float(np.sqrt(np.mean(shaped.astype(np.float64) ** 2) + 1e-12))
    if target_rms > 1e-7 and shaped_rms > 1e-8:
        shaped *= float(np.clip(target_rms / shaped_rms, 10 ** (-0.8 / 20), 10 ** (0.8 / 20)))

    fade = min(len(target) // 4, int(round(0.008 * sr)))
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
        "trajectory_contrast": float(contrast),
        "trajectory_revision": 2,
        "voiced_onset_zero": bool(voiced_onset_zero),
    }


def transfer_articulation_trajectory_v2(
    source,
    target,
    sr,
    strength=0.82,
    transition_fraction=0.35,
    voiced_onset_zero=False,
):
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    template = extract_neutral_articulation_template(source, sr)
    if template is None:
        return target.copy(), False, {
            "trajectory_gain_rms_db": 0.0,
            "trajectory_strength": 0.0,
            "canonical_coherence": 0.0,
            "trajectory_contrast": 0.0,
            "trajectory_revision": 2,
            "voiced_onset_zero": bool(voiced_onset_zero),
        }
    template["coherence"] = 0.55
    template["source_count"] = 1
    return apply_articulation_template_v2(
        template,
        target,
        sr,
        strength=strength,
        transition_fraction=transition_fraction,
        voiced_onset_zero=voiced_onset_zero,
    )


def single_source_articulation_hybrid_v2(
    original,
    generated,
    sr,
    regions,
    source_fixed_ms,
    target_fixed_ms,
    target_ms,
    canonical_template=None,
):
    original = np.asarray(original, dtype=np.float32)
    generated = np.asarray(generated, dtype=np.float32)
    n = len(generated)
    source_total_ms = len(original) * 1000.0 / float(sr)
    mapped = map_articulation_regions(
        regions,
        source_fixed_ms,
        target_fixed_ms,
        source_total_ms,
        target_ms,
    )

    raw_src_ms = float(regions.get("raw_end_ms", max(0.0, regions.get("first_voiced_ms", 0.0) - 5.0)))
    onset_src_ms = float(regions.get("first_voiced_ms", 0.0))
    transition_src_ms = float(regions.get("transition_end_ms", onset_src_ms + 70.0))
    end_src_ms = float(regions.get("articulation_end_ms", onset_src_ms + 120.0))

    raw_src = int(np.clip(round(raw_src_ms * sr / 1000.0), 0, len(original)))
    onset_src = int(np.clip(round(onset_src_ms * sr / 1000.0), raw_src, len(original)))
    end_src = int(np.clip(round(end_src_ms * sr / 1000.0), onset_src, len(original)))

    raw_tgt = int(np.clip(round(mapped["target_raw_end_ms"] * sr / 1000.0), 0, n))
    onset_tgt = int(np.clip(round(mapped["target_onset_ms"] * sr / 1000.0), raw_tgt, n))
    transition_tgt = int(np.clip(round(mapped["target_transition_end_ms"] * sr / 1000.0), onset_tgt, n))
    end_tgt = int(np.clip(round(mapped["target_articulation_end_ms"] * sr / 1000.0), transition_tgt, n))

    span = max(1, end_tgt - onset_tgt)
    transition_fraction = float(np.clip((transition_tgt - onset_tgt) / span, 0.10, 0.78))
    voiced_onset_zero = bool(onset_src_ms <= 8.0 and transition_src_ms >= 24.0)

    out = generated.copy()
    trajectory_used = False
    source_kind = "none"
    trajectory_stats = {
        "trajectory_gain_rms_db": 0.0,
        "trajectory_strength": 0.0,
        "canonical_coherence": 0.0,
        "trajectory_contrast": 0.0,
        "trajectory_revision": 2,
        "voiced_onset_zero": voiced_onset_zero,
    }

    if end_tgt - onset_tgt >= 256:
        confidence = float(np.clip(regions.get("confidence", 0.7), 0.0, 1.0))
        if canonical_template is not None:
            strength = 0.74 + 0.14 * confidence
            shaped, trajectory_used, trajectory_stats = apply_articulation_template_v2(
                canonical_template,
                out[onset_tgt:end_tgt],
                sr,
                strength=strength,
                transition_fraction=transition_fraction,
                voiced_onset_zero=voiced_onset_zero,
            )
            source_kind = "canonical"
        elif end_src - onset_src >= 256:
            strength = 0.62 + 0.16 * confidence
            shaped, trajectory_used, trajectory_stats = transfer_articulation_trajectory_v2(
                original[onset_src:end_src],
                out[onset_tgt:end_tgt],
                sr,
                strength=strength,
                transition_fraction=transition_fraction,
                voiced_onset_zero=voiced_onset_zero,
            )
            source_kind = "neutralized_local"
        else:
            shaped = out[onset_tgt:end_tgt]
        if trajectory_used:
            out[onset_tgt:end_tgt] = shaped

    # Keep the proven v1 behavior for genuinely unvoiced material. Voiced onset
    # material is never copied raw, preventing source-pitch leakage.
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
