#!/usr/bin/env python3
import json
import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf


PROFILE_FORMAT = 3
BAND_EDGES = np.asarray([8000.0, 10000.0, 12000.0, 14000.0, 16000.0, 18000.0, 20000.0], dtype=np.float32)
BAND_CENTERS = 0.5 * (BAND_EDGES[:-1] + BAND_EDGES[1:])
TEMP_FIXED_BINS = 16
TEMP_TAIL_BINS = 32


def _mono(audio):
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.nan_to_num(np.asarray(audio, dtype=np.float32))


def crop_native(audio, sr, offset_ms, cutoff_ms):
    audio = np.asarray(audio, dtype=np.float32)
    start = int(round(max(0.0, float(offset_ms)) * sr / 1000.0))
    if float(cutoff_ms) < 0.0:
        end = start + int(round((-float(cutoff_ms)) * sr / 1000.0))
    else:
        end = len(audio) - int(round(max(0.0, float(cutoff_ms)) * sr / 1000.0))
    start = int(np.clip(start, 0, len(audio)))
    end = int(np.clip(end, start, len(audio)))
    return audio[start:end]


def _frame_band_rms(audio, sr, n_fft=4096, hop=512):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size < 512:
        return None
    n_fft = int(min(n_fft, 2 ** int(np.floor(np.log2(max(512, audio.size))))))
    n_fft = max(512, n_fft)
    hop = max(128, min(hop, n_fft // 4))
    pad = n_fft // 2
    padded = np.pad(audio, (pad, pad), mode="reflect" if audio.size > 1 else "constant")
    window = np.hanning(n_fft).astype(np.float64)
    frames = []
    for start in range(0, max(1, padded.size - n_fft + 1), hop):
        frame = padded[start:start + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size))
        spec = np.fft.rfft(frame.astype(np.float64) * window)
        frames.append(np.abs(spec) ** 2)
    if not frames:
        return None
    power = np.stack(frames, axis=1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    total = np.maximum(np.sum(power, axis=0), 1e-18)
    band = np.zeros((len(BAND_CENTERS), power.shape[1]), dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(BAND_EDGES[:-1], BAND_EDGES[1:])):
        mask = (freqs >= lo) & (freqs < min(float(hi), sr * 0.5))
        if np.any(mask):
            band[i] = np.sqrt(np.sum(power[mask], axis=0) / total)
    high_mask = (freqs >= 12000.0) & (freqs < min(20000.0, sr * 0.5))
    flatness = np.ones(power.shape[1], dtype=np.float64)
    if np.any(high_mask):
        hp = np.maximum(power[high_mask], 1e-18)
        flatness = np.exp(np.mean(np.log(hp), axis=0)) / np.maximum(np.mean(hp, axis=0), 1e-18)
    times = np.arange(power.shape[1], dtype=np.float64) * hop / float(sr)
    return band, np.clip(flatness, 0.0, 1.0), times


def _piecewise_resample(values, times, fixed_sec, duration_sec, fixed_bins=TEMP_FIXED_BINS, tail_bins=TEMP_TAIL_BINS):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(fixed_bins + tail_bins, dtype=np.float64)
    if times.size != values.size:
        times = np.linspace(0.0, max(duration_sec, 1e-6), values.size)
    duration_sec = max(float(duration_sec), 1e-6)
    fixed_sec = float(np.clip(fixed_sec, 0.0, duration_sec))
    fallback = float(np.median(values))

    def region(a, b, bins):
        if bins <= 0:
            return np.zeros(0, dtype=np.float64)
        if b <= a + 1e-6:
            return np.full(bins, fallback, dtype=np.float64)
        mask = (times >= a - 1e-9) & (times <= b + 1e-9)
        if np.sum(mask) < 2:
            return np.full(bins, fallback, dtype=np.float64)
        tx = (times[mask] - a) / max(b - a, 1e-9)
        return np.interp(np.linspace(0.0, 1.0, bins), tx, values[mask], left=values[mask][0], right=values[mask][-1])

    return np.concatenate([
        region(0.0, fixed_sec, fixed_bins),
        region(fixed_sec, duration_sec, tail_bins),
    ])


def _trajectory_from_frames(band, flatness, times, voiced_f, fixed_sec, duration_sec):
    low_amp = np.sqrt(np.mean(np.maximum(band[:2], 1e-8) ** 2, axis=0))
    upper_amp = np.sqrt(np.mean(np.maximum(band[2:], 1e-8) ** 2, axis=0))
    low_ref = max(float(np.median(low_amp)), 1e-6)
    upper_ref = max(float(np.median(upper_amp)), 1e-6)
    low_delta = np.clip(20.0 * np.log10(np.maximum(low_amp, 1e-7) / low_ref), -16.0, 16.0)
    upper_delta = np.clip(20.0 * np.log10(np.maximum(upper_amp, 1e-7) / upper_ref), -16.0, 16.0)
    harmonic_mix = np.clip(1.15 - 1.35 * np.asarray(flatness, dtype=np.float64), 0.12, 0.95)
    voicing = np.asarray(voiced_f, dtype=np.float64)
    return {
        "fixed_bins": TEMP_FIXED_BINS,
        "tail_bins": TEMP_TAIL_BINS,
        "low_delta_db": [float(x) for x in _piecewise_resample(low_delta, times, fixed_sec, duration_sec)],
        "upper_delta_db": [float(x) for x in _piecewise_resample(upper_delta, times, fixed_sec, duration_sec)],
        "harmonic_mix": [float(x) for x in _piecewise_resample(harmonic_mix, times, fixed_sec, duration_sec)],
        "voicing": [float(x) for x in _piecewise_resample(voicing, times, fixed_sec, duration_sec)],
        "upper_peak_delta_db": float(np.clip(np.percentile(upper_delta, 90), 0.0, 16.0)),
        "low_peak_delta_db": float(np.clip(np.percentile(low_delta, 90), 0.0, 16.0)),
    }


def _limit_band_slope(db_values, max_rise_db=4.5, max_drop_db=7.0):
    x = np.asarray(db_values, dtype=np.float64).copy()
    if x.size < 2:
        return x
    for i in range(1, x.size):
        x[i] = np.clip(x[i], x[i - 1] - max_drop_db, x[i - 1] + max_rise_db)
    for i in range(x.size - 2, -1, -1):
        x[i] = np.clip(x[i], x[i + 1] - max_rise_db, x[i + 1] + max_drop_db)
    if x.size >= 3:
        y = x.copy()
        y[1:-1] = 0.18 * x[:-2] + 0.64 * x[1:-1] + 0.18 * x[2:]
        x = y
    return x


def analyze_sample_profile(wav_path, offset_ms, cutoff_ms, consonant_ms, f0_frames, model_hop=256, model_sr=24000):
    audio, sr = sf.read(wav_path, always_2d=False)
    audio = crop_native(_mono(audio), int(sr), offset_ms, cutoff_ms)
    sr = int(sr)
    if audio.size < int(0.06 * sr) or sr * 0.5 < 13500.0:
        return None
    measured = _frame_band_rms(audio, sr)
    if measured is None:
        return None
    band, flatness, times = measured
    f0_frames = np.asarray(f0_frames, dtype=np.float32).reshape(-1)
    if f0_frames.size:
        mt = np.arange(f0_frames.size, dtype=np.float64) * float(model_hop) / float(model_sr)
        voiced_f = np.interp(times, mt, (f0_frames > 1.0).astype(np.float64), left=0.0, right=0.0) >= 0.5
    else:
        voiced_f = np.zeros(times.size, dtype=bool)
    unvoiced_f = ~voiced_f
    if np.sum(voiced_f) < 2:
        voiced_f[:] = True
    if np.sum(unvoiced_f) < 2:
        unvoiced_f[:] = True

    def profile_for(mask):
        vals = np.median(band[:, mask], axis=1) if np.any(mask) else np.median(band, axis=1)
        ratios = np.clip(vals, 1e-4, 0.60)
        return _limit_band_slope(20.0 * np.log10(ratios)).astype(np.float32)

    voiced_db = profile_for(voiced_f)
    unvoiced_db = profile_for(unvoiced_f)
    voiced_flatness = float(np.median(flatness[voiced_f])) if np.any(voiced_f) else float(np.median(flatness))
    harmonic_mix = float(np.clip(1.15 - 1.35 * voiced_flatness, 0.20, 0.92))
    duration_sec = audio.size / float(sr)
    fixed_sec = min(max(0.0, float(consonant_ms)) / 1000.0, duration_sec)
    return {
        "format": PROFILE_FORMAT,
        "source_sr": sr,
        "source_nyquist_hz": sr * 0.5,
        "band_centers_hz": [float(x) for x in BAND_CENTERS],
        "voiced_db_to_full": [float(x) for x in voiced_db],
        "unvoiced_db_to_full": [float(x) for x in unvoiced_db],
        "voiced_harmonic_mix": harmonic_mix,
        "edge_voiced_db": float(np.mean(voiced_db[:2])),
        "edge_unvoiced_db": float(np.mean(unvoiced_db[:2])),
        "source_fixed_ratio": float(fixed_sec / max(duration_sec, 1e-6)),
        "temporal": _trajectory_from_frames(band, flatness, times, voiced_f, fixed_sec, duration_sec),
    }


def _average_profiles(profiles):
    profiles = [p for p in profiles if p]
    if not profiles:
        return None
    voiced_db = np.asarray([p["voiced_db_to_full"] for p in profiles], dtype=np.float64)
    unvoiced_db = np.asarray([p["unvoiced_db_to_full"] for p in profiles], dtype=np.float64)
    voiced = 20.0 * np.log10(np.maximum(np.median(np.power(10.0, voiced_db / 20.0), axis=0), 1e-6))
    unvoiced = 20.0 * np.log10(np.maximum(np.median(np.power(10.0, unvoiced_db / 20.0), axis=0), 1e-6))
    voiced = _limit_band_slope(voiced)
    unvoiced = _limit_band_slope(unvoiced)
    out = {
        "band_centers_hz": [float(x) for x in BAND_CENTERS],
        "voiced_db_to_full": [float(x) for x in np.clip(voiced, -80, -2)],
        "unvoiced_db_to_full": [float(x) for x in np.clip(unvoiced, -80, -2)],
        "voiced_harmonic_mix": float(np.median([p.get("voiced_harmonic_mix", 0.65) for p in profiles])),
        "source_nyquist_hz": float(max(p.get("source_nyquist_hz", 0.0) for p in profiles)),
        "sample_count": len(profiles),
        "edge_voiced_db": float(np.mean(voiced[:2])),
        "edge_unvoiced_db": float(np.mean(unvoiced[:2])),
    }
    temporal = [p.get("temporal") for p in profiles if isinstance(p.get("temporal"), dict)]
    if temporal:
        def avg_delta(key):
            arr = np.asarray([t[key] for t in temporal], dtype=np.float64)
            return 20.0 * np.log10(np.maximum(np.median(np.power(10.0, arr / 20.0), axis=0), 1e-6))

        def avg_linear(key):
            return np.median(np.asarray([t[key] for t in temporal], dtype=np.float64), axis=0)

        out["temporal"] = {
            "fixed_bins": TEMP_FIXED_BINS,
            "tail_bins": TEMP_TAIL_BINS,
            "low_delta_db": [float(x) for x in np.clip(avg_delta("low_delta_db"), -16, 16)],
            "upper_delta_db": [float(x) for x in np.clip(avg_delta("upper_delta_db"), -16, 16)],
            "harmonic_mix": [float(x) for x in np.clip(avg_linear("harmonic_mix"), 0.12, 0.95)],
            "voicing": [float(x) for x in np.clip(avg_linear("voicing"), 0.0, 1.0)],
            "upper_peak_delta_db": float(np.median([t.get("upper_peak_delta_db", 0.0) for t in temporal])),
            "low_peak_delta_db": float(np.median([t.get("low_peak_delta_db", 0.0) for t in temporal])),
        }
    return out


def build_profile_database(voicebank_root, manifest_entries, model_hop=256, model_sr=24000, state_dir=None):
    root = Path(voicebank_root).resolve()
    state_dir = Path(state_dir).resolve() if state_dir is not None else root / ".yuaz-alpha8-rc3-2"
    cache_dir = state_dir / "highband_cache_v3_ai14"
    cache_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    analyzed = cached = skipped = 0
    for item in manifest_entries:
        if item.get("status") == "error" or not item.get("cache"):
            continue
        base_alias = str(item.get("base_alias") or item.get("alias") or "").strip()
        if not base_alias:
            continue
        wav = Path(item.get("wav_path") or (root / item.get("relative_wav", ""))).expanduser()
        wav = wav if wav.is_absolute() else root / wav
        if not wav.exists():
            skipped += 1
            continue
        try:
            with np.load(root / item["cache"], allow_pickle=False) as data:
                f0 = np.asarray(data["f0"], dtype=np.float32)
        except Exception:
            skipped += 1
            continue
        import hashlib
        signature = str(item.get("loudness_signature") or "")
        key_src = f"{item.get('sha256','')}|{signature}|c{float(item.get('consonant',0.0)):.3f}|hb{PROFILE_FORMAT}"
        hp = cache_dir / (hashlib.sha1(key_src.encode("utf-8", "replace")).hexdigest()[:24] + ".json")
        profile = None
        if hp.exists():
            try:
                profile = json.loads(hp.read_text(encoding="utf-8"))
                if int(profile.get("format", 0)) == PROFILE_FORMAT:
                    cached += 1
                else:
                    profile = None
            except Exception:
                profile = None
        if profile is None:
            try:
                profile = analyze_sample_profile(
                    wav, item.get("offset", 0.0), item.get("cutoff", 0.0), item.get("consonant", 0.0),
                    f0, model_hop, model_sr,
                )
            except Exception:
                profile = None
            if profile:
                hp.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
                analyzed += 1
        if not profile:
            skipped += 1
            continue
        idx = int(item.get("subbank_index", 0) or 0)
        label = str(item.get("subbank_label") or "default")
        anchor = float(item.get("subbank_anchor_midi", 60.0) or 60.0)
        slot = groups.setdefault(base_alias, {}).setdefault(idx, {
            "subbank_index": idx,
            "subbank_label": label,
            "anchor_midi": anchor,
            "profiles": [],
        })
        slot["profiles"].append(profile)

    out_groups = {}
    temporal_prototypes = 0
    for alias, slots in groups.items():
        protos = []
        for _, slot in sorted(slots.items()):
            agg = _average_profiles(slot["profiles"])
            if not agg:
                continue
            if agg.get("temporal"):
                temporal_prototypes += 1
            protos.append({
                "subbank_index": int(slot["subbank_index"]),
                "subbank_label": slot["subbank_label"],
                "anchor_midi": float(slot["anchor_midi"]),
                **agg,
            })
        if protos:
            out_groups[alias] = {"prototypes": protos}
    return {
        "format": PROFILE_FORMAT,
        "voicebank_root": str(root),
        "band_centers_hz": [float(x) for x in BAND_CENTERS],
        "temporal_layout": {"fixed_bins": TEMP_FIXED_BINS, "tail_bins": TEMP_TAIL_BINS},
        "continuity_mode": "source_texture_nonlinear_v2",
        "synthesis_mode": "bandlimited_source_texture",
        "groups": out_groups,
        "stats": {
            "analyzed": analyzed,
            "cached": cached,
            "skipped": skipped,
            "alias_count": len(out_groups),
            "temporal_prototypes": temporal_prototypes,
        },
    }


def save_profile_database(path, db):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8")


def _prepare_bank_fallback(data):
    groups = data.get("groups") or {}
    slots = {}
    for group in groups.values():
        for proto in (group or {}).get("prototypes") or []:
            idx = int(proto.get("subbank_index", 0) or 0)
            slots.setdefault(idx, []).append(proto)
    protos = []
    for idx, items in sorted(slots.items()):
        agg = _average_profiles(items)
        if not agg:
            continue
        anchors = [float(x.get("anchor_midi", 60.0) or 60.0) for x in items]
        protos.append({
            "subbank_index": idx,
            "subbank_label": "bank-wide",
            "anchor_midi": float(np.median(anchors)) if anchors else 60.0,
            **agg,
        })
    data["_bank_fallback"] = {"prototypes": protos}
    return data


def load_profile_database(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(data.get("format", 0)) != PROFILE_FORMAT:
        raise RuntimeError(f"Unsupported high-band profile format: {data.get('format')}")
    return _prepare_bank_fallback(data)


def _profile_weights(prototypes, target_midi, source_prototype_index=None):
    anchors = np.asarray([float(p.get("anchor_midi", 60.0)) for p in prototypes], dtype=np.float64)
    if anchors.size == 1:
        return np.ones(1, dtype=np.float64)
    spacing = np.diff(np.sort(anchors))
    spacing = spacing[spacing > 0.25]
    sigma = float(np.median(spacing) * 0.60) if spacing.size else 3.6
    sigma = float(np.clip(sigma, 1.6, 5.2))
    logits = -0.5 * ((float(target_midi) - anchors) / sigma) ** 2
    if source_prototype_index is not None:
        for j, proto in enumerate(prototypes):
            if int(proto.get("subbank_index", -999)) == int(source_prototype_index):
                logits[j] += 1.55 * math.exp(-0.5 * ((float(target_midi) - anchors[j]) / 8.0) ** 2)
                break
    logits -= np.max(logits)
    weights = np.exp(logits)
    return weights / max(float(np.sum(weights)), 1e-12)


def _norm_alias(value):
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def select_learned_profile(db, base_alias, target_midi, timbre_shift_semitones=0.0, source_prototype_index=None):
    if not db:
        return None
    groups = db.get("groups") or {}
    requested = str(base_alias or "")
    selected_alias = None
    match_mode = "exact"
    group = groups.get(requested) if requested else None
    if not group and requested:
        wanted = _norm_alias(requested)
        for alias, candidate in groups.items():
            if _norm_alias(alias) == wanted:
                group = candidate
                selected_alias = str(alias)
                match_mode = "normalized-alias"
                break
    if group and selected_alias is None:
        selected_alias = requested
    protos = (group or {}).get("prototypes") or []
    if not protos:
        fallback = db.get("_bank_fallback")
        if not fallback:
            fallback = _prepare_bank_fallback(db).get("_bank_fallback")
        protos = (fallback or {}).get("prototypes") or []
        selected_alias = "<bank-wide>"
        match_mode = "bank-wide-fallback"
    if not protos:
        return None
    weights = _profile_weights(protos, float(target_midi) + float(timbre_shift_semitones), source_prototype_index)
    voiced_db = np.asarray([p["voiced_db_to_full"] for p in protos], dtype=np.float64)
    unvoiced_db = np.asarray([p["unvoiced_db_to_full"] for p in protos], dtype=np.float64)
    voiced_amp = np.sum(np.power(10.0, voiced_db / 20.0) * weights[:, None], axis=0)
    unvoiced_amp = np.sum(np.power(10.0, unvoiced_db / 20.0) * weights[:, None], axis=0)
    voiced_mix_db = _limit_band_slope(20.0 * np.log10(np.maximum(voiced_amp, 1e-6)))
    unvoiced_mix_db = _limit_band_slope(20.0 * np.log10(np.maximum(unvoiced_amp, 1e-6)))
    harmonic_mix = float(np.sum(np.asarray([p.get("voiced_harmonic_mix", 0.65) for p in protos]) * weights))
    out = {
        "band_centers_hz": [float(x) for x in BAND_CENTERS],
        "voiced_db_to_full": [float(x) for x in voiced_mix_db],
        "unvoiced_db_to_full": [float(x) for x in unvoiced_mix_db],
        "voiced_harmonic_mix": float(np.clip(harmonic_mix, 0.15, 0.95)),
        "edge_voiced_db": float(np.mean(voiced_mix_db[:2])),
        "edge_unvoiced_db": float(np.mean(unvoiced_mix_db[:2])),
        "weights": [float(x) for x in weights],
        "match_mode": match_mode,
        "requested_base_alias": requested,
        "selected_base_alias": selected_alias,
    }
    temporals = [p.get("temporal") for p in protos]
    if all(isinstance(t, dict) for t in temporals):
        def mix_delta(key):
            arr = np.asarray([t[key] for t in temporals], dtype=np.float64)
            return 20.0 * np.log10(np.maximum(np.sum(np.power(10.0, arr / 20.0) * weights[:, None], axis=0), 1e-6))

        def mix_linear(key):
            return np.sum(np.asarray([t[key] for t in temporals], dtype=np.float64) * weights[:, None], axis=0)

        out["temporal"] = {
            "fixed_bins": TEMP_FIXED_BINS,
            "tail_bins": TEMP_TAIL_BINS,
            "low_delta_db": [float(x) for x in np.clip(mix_delta("low_delta_db"), -16, 16)],
            "upper_delta_db": [float(x) for x in np.clip(mix_delta("upper_delta_db"), -16, 16)],
            "harmonic_mix": [float(x) for x in np.clip(mix_linear("harmonic_mix"), 0.12, 0.95)],
            "voicing": [float(x) for x in np.clip(mix_linear("voicing"), 0.0, 1.0)],
            "upper_peak_delta_db": float(np.sum(np.asarray([t.get("upper_peak_delta_db", 0.0) for t in temporals]) * weights)),
            "low_peak_delta_db": float(np.sum(np.asarray([t.get("low_peak_delta_db", 0.0) for t in temporals]) * weights)),
        }
    return out


def _smooth_mask(mask, samples):
    x = np.asarray(mask, dtype=np.float32)
    samples = max(1, int(samples))
    if samples <= 1 or x.size < 4:
        return np.clip(x, 0.0, 1.0)
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    idx = np.arange(x.size)
    lo = np.maximum(0, idx - samples // 2)
    hi = np.minimum(x.size, idx + samples // 2 + 1)
    return np.clip((c[hi] - c[lo]) / np.maximum(1, hi - lo), 0.0, 1.0).astype(np.float32)


def _moving_rms(audio, samples):
    x2 = np.asarray(audio, dtype=np.float64) ** 2
    samples = max(1, int(samples))
    c = np.concatenate([[0.0], np.cumsum(x2)])
    idx = np.arange(x2.size)
    lo = np.maximum(0, idx - samples // 2)
    hi = np.minimum(x2.size, idx + samples // 2 + 1)
    return np.sqrt((c[hi] - c[lo]) / np.maximum(1, hi - lo) + 1e-12).astype(np.float32)


def _band_curve(freqs, centers, db, low_cut):
    centers = np.asarray(centers, dtype=np.float64)
    db = _limit_band_slope(np.asarray(db, dtype=np.float64))
    amp = np.power(10.0, db / 20.0)
    curve = np.interp(freqs, centers, amp, left=amp[0], right=amp[-1])
    x = np.clip((freqs - float(low_cut)) / 900.0, 0.0, 1.0)
    curve *= x * x * (3.0 - 2.0 * x)
    curve[freqs > 20000.0] = 0.0
    return curve.astype(np.float64)


def _render_temporal(profile, n_samples, sr, target_fixed_ms):
    temporal = profile.get("temporal") if isinstance(profile, dict) else None
    if not isinstance(temporal, dict):
        one = np.ones(n_samples, dtype=np.float64)
        return one, one, np.full(n_samples, float(profile.get("voiced_harmonic_mix", 0.65)), dtype=np.float64)
    fixed_bins = int(temporal.get("fixed_bins", TEMP_FIXED_BINS))
    tail_bins = int(temporal.get("tail_bins", TEMP_TAIL_BINS))
    fixed_n = int(np.clip(round(float(target_fixed_ms) * sr / 1000.0), 0, n_samples))

    def expand(values, default):
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        if vals.size < fixed_bins + tail_bins:
            return np.full(n_samples, default, dtype=np.float64)
        out = np.empty(n_samples, dtype=np.float64)
        if fixed_n > 0:
            out[:fixed_n] = np.interp(np.linspace(0, 1, fixed_n), np.linspace(0, 1, fixed_bins), vals[:fixed_bins])
        if fixed_n < n_samples:
            out[fixed_n:] = np.interp(np.linspace(0, 1, n_samples - fixed_n), np.linspace(0, 1, tail_bins), vals[fixed_bins:fixed_bins + tail_bins])
        return out

    low = np.power(10.0, np.clip(expand(temporal.get("low_delta_db", []), 0.0), -16, 16) / 20.0)
    upper = np.power(10.0, np.clip(expand(temporal.get("upper_delta_db", []), 0.0), -16, 16) / 20.0)
    harmonic_mix = np.clip(expand(temporal.get("harmonic_mix", []), float(profile.get("voiced_harmonic_mix", 0.65))), 0.12, 0.95)
    smooth = max(1, int(0.006 * sr))

    def fast_smooth(x):
        if smooth <= 1 or x.size < 4:
            return x
        c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
        idx = np.arange(x.size)
        lo = np.maximum(0, idx - smooth // 2)
        hi = np.minimum(x.size, idx + smooth // 2 + 1)
        return (c[hi] - c[lo]) / np.maximum(1, hi - lo)

    return (
        np.clip(fast_smooth(low), 0.24, 4.0),
        np.clip(fast_smooth(upper), 0.24, 4.0),
        np.clip(fast_smooth(harmonic_mix), 0.12, 0.95),
    )


def _edge_ratio(audio, sr, low=8200.0, high=11600.0):
    x = np.asarray(audio, dtype=np.float64)
    if x.size < 512:
        return 0.0
    n = min(x.size, 32768)
    segment = x[(x.size - n) // 2:(x.size - n) // 2 + n]
    window = np.hanning(n)
    power = np.abs(np.fft.rfft(segment * window)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    total = float(np.sum(power)) + 1e-18
    mask = (freqs >= low) & (freqs < min(high, sr * 0.5))
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.sum(power[mask]) / total))


def _band_ratio(audio, sr, low, high):
    return _edge_ratio(audio, sr, low=float(low), high=float(high))


def synthesize_learned_highband(generated, sr, target_f0, profile, seed, assist_start_hz=10000.0, detail_strength=1.0, target_fixed_ms=0.0, restoration_strength=1.0):
    y = np.asarray(generated, dtype=np.float32)
    sr = int(sr)
    strength = float(np.clip(restoration_strength, 0.0, 1.0))
    if strength <= 1e-8:
        return y.copy(), {"used": False, "reason": "zero-strength", "restoration_strength": 0.0}
    if profile is None or y.size < 512 or sr * 0.5 < 14000.0:
        return y.copy(), {"used": False, "reason": "no-profile-or-bandwidth"}
    f0_frames = np.asarray(target_f0, dtype=np.float32).reshape(-1)
    if f0_frames.size == 0:
        return y.copy(), {"used": False, "reason": "no-f0"}

    f0 = np.interp(np.linspace(0, 1, y.size), np.linspace(0, 1, f0_frames.size), f0_frames.astype(np.float64))
    voiced = _smooth_mask((f0 > 1.0).astype(np.float32), int(0.012 * sr))
    unvoiced = 1.0 - voiced
    f0_safe = np.where(f0 > 1.0, f0, np.median(f0[f0 > 1.0]) if np.any(f0 > 1.0) else 220.0)
    envelope = _moving_rms(y, int(0.014 * sr))
    body_rms = max(float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-12)), 1e-6)
    envelope = np.clip(envelope / body_rms, 0.10, 3.0)
    low_t, upper_t, harmonic_mix_t = _render_temporal(profile, y.size, sr, target_fixed_ms)

    centers = np.asarray(profile["band_centers_hz"], dtype=np.float64)
    voiced_db = _limit_band_slope(np.asarray(profile["voiced_db_to_full"], dtype=np.float64))
    unvoiced_db = _limit_band_slope(np.asarray(profile["unvoiced_db_to_full"], dtype=np.float64))
    voiced_amp = np.power(10.0, voiced_db / 20.0)
    unvoiced_amp = np.power(10.0, unvoiced_db / 20.0)
    harmonic_mix = float(np.clip(profile.get("voiced_harmonic_mix", 0.65), 0.15, 0.95))

    # Anchor reconstruction to the rendered upper-body edge when source profiles are sparse.
    learned_edge = max(float(np.mean(voiced_amp[:2])), 1e-5)
    actual_edge = _edge_ratio(y, sr)
    lower_anchor = _band_ratio(y, sr, 6500.0, 10000.0)
    edge_ratio = actual_edge / learned_edge if actual_edge > 0.0 else 1.0
    continuity_gain = float(np.clip(0.72 + 0.28 * math.sqrt(max(edge_ratio, 1e-5)), 0.72, 1.34))
    upper_continuity = np.linspace(1.0, continuity_gain, len(voiced_amp))
    voiced_amp = voiced_amp * upper_continuity
    unvoiced_amp = unvoiced_amp * np.linspace(1.0, float(np.clip(continuity_gain, 0.78, 1.20)), len(unvoiced_amp))

    anchor_floor = float(np.clip(max(actual_edge, lower_anchor * 0.42) * 0.30, 0.0025, 0.040))
    voiced_floor = anchor_floor * np.asarray([1.15, 1.00, 0.82, 0.66, 0.52, 0.40], dtype=np.float64)
    unvoiced_floor = anchor_floor * np.asarray([1.30, 1.18, 1.02, 0.86, 0.70, 0.56], dtype=np.float64)
    voiced_amp = np.maximum(voiced_amp, voiced_floor)
    unvoiced_amp = np.maximum(unvoiced_amp, unvoiced_floor)

    # Derive upper-band excitation from the rendered signal with quadratic/cubic nonlinear paths.
    bridge_start = max(8600.0, min(11400.0, float(assist_start_hz) + 250.0))
    bridge_full = min(12400.0, bridge_start + 1500.0)
    max_freq = min(20000.0, sr * 0.5 - 300.0)
    fft_freqs = np.fft.rfftfreq(y.size, d=1.0 / float(sr))
    y_spec = np.fft.rfft(y.astype(np.float64))

    def smooth_lowpass(cutoff, transition=900.0):
        lo = max(20.0, float(cutoff) - float(transition))
        x = np.clip((fft_freqs - lo) / max(1.0, float(transition)), 0.0, 1.0)
        fall = 1.0 - (x * x * (3.0 - 2.0 * x))
        return np.fft.irfft(y_spec * fall, n=y.size).real

    # Band-limit nonlinear paths to prevent Nyquist fold-back.
    quad_src = smooth_lowpass(min(9400.0, max_freq * 0.49), transition=850.0)
    cubic_src = smooth_lowpass(min(6300.0, max_freq * 0.325), transition=750.0)
    quad_n = np.clip(quad_src / body_rms, -3.5, 3.5)
    cubic_n = np.clip(cubic_src / body_rms, -3.0, 3.0)
    quad = quad_n * quad_n
    quad -= float(np.mean(quad))
    cubic = cubic_n * cubic_n * cubic_n - 0.72 * cubic_n
    exciter = 0.58 * quad + 0.42 * cubic

    exc_spec = np.fft.rfft(exciter)
    # Add narrow deterministic skirts before voicebank-envelope shaping.
    bin_hz = float(sr) / max(1, y.size)
    skirt_bins = max(3, int(round(48.0 / max(bin_hz, 1e-6))))
    power = np.abs(exc_spec) ** 2
    kernel = np.ones(skirt_bins, dtype=np.float64) / float(skirt_bins)
    smooth_power = np.convolve(power, kernel, mode="same")
    texture_rng = np.random.default_rng((int(seed) ^ 0x6A09E667) & 0xFFFFFFFF)
    skirt_phase = texture_rng.uniform(-np.pi, np.pi, exc_spec.size)
    skirt_spec = np.sqrt(np.maximum(smooth_power, 0.0)) * np.exp(1j * skirt_phase)
    skirt_mix = float(np.clip(0.18 + 0.28 * (1.0 - harmonic_mix), 0.18, 0.38))
    exc_spec = (1.0 - skirt_mix) * exc_spec + skirt_mix * skirt_spec
    rise_x = np.clip((fft_freqs - bridge_start) / max(250.0, bridge_full - bridge_start), 0.0, 1.0)
    highpass = rise_x * rise_x * (3.0 - 2.0 * rise_x)
    fall_start = max(17500.0, max_freq - 1800.0)
    fall_x = np.clip((fft_freqs - fall_start) / max(300.0, max_freq - fall_start), 0.0, 1.0)
    lowpass = 1.0 - (fall_x * fall_x * (3.0 - 2.0 * fall_x))
    lowpass[fft_freqs >= max_freq] = 0.0

    profile_shape = np.interp(fft_freqs, centers, voiced_amp, left=voiced_amp[0], right=voiced_amp[-1])
    hb_mask = (fft_freqs >= bridge_start) & (fft_freqs <= max_freq)
    profile_ref = float(np.mean(profile_shape[hb_mask])) if np.any(hb_mask) else 1.0
    profile_shape = np.clip(profile_shape / max(profile_ref, 1e-7), 0.24, 3.2)
    frequency_tilt = np.power(np.maximum(fft_freqs, bridge_start) / max(bridge_start, 1.0), -0.32)
    texture_spec = exc_spec * highpass * lowpass * profile_shape * frequency_tilt
    harmonic = np.fft.irfft(texture_spec, n=y.size).real

    voiced_body = float(np.sqrt(np.mean((y * voiced) ** 2) + 1e-12))
    voiced_f0 = f0[f0 > 1.0]
    median_f0 = float(np.median(voiced_f0)) if voiced_f0.size else 220.0
    low_pitch_comp = float(np.clip((180.0 / max(median_f0, 45.0)) ** 0.22, 0.88, 1.48))
    target_harmonic = voiced_body * float(np.mean(voiced_amp[2:])) * harmonic_mix * low_pitch_comp
    harmonic *= voiced * envelope * upper_t * np.sqrt(np.clip(harmonic_mix_t / max(harmonic_mix, 1e-4), 0.35, 2.2))
    harmonic_rms = float(np.sqrt(np.mean(harmonic ** 2) + 1e-12))
    if harmonic_rms > 1e-9:
        harmonic *= min(7.0, target_harmonic / harmonic_rms)

    # Diagnostic estimate; the texture path has no fixed partial-count ceiling.
    harmonic_count = max(0, int(max_freq / max(median_f0, 1.0)) - int(bridge_start / max(median_f0, 1.0)))

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    noise = rng.standard_normal(y.size).astype(np.float64)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(y.size, d=1.0 / float(sr))
    noise_low_cut = max(8200.0, min(11600.0, float(assist_start_hz)))
    noise_curve = _band_curve(freqs, centers, 20.0 * np.log10(np.maximum(unvoiced_amp, 1e-7)), noise_low_cut)
    shaped = np.fft.irfft(spec * noise_curve, n=y.size).real
    start_weight = np.clip((12000.0 - float(assist_start_hz)) / 4000.0, 0.0, 1.0)
    noise_time = low_t * start_weight + upper_t * (1.0 - start_weight)
    voiced_air = 0.10 + 0.42 * (1.0 - harmonic_mix_t)
    noise_mask = unvoiced + voiced * voiced_air
    noise_rms = float(np.sqrt(np.mean((shaped * noise_mask) ** 2) + 1e-12))
    unvoiced_body = float(np.sqrt(np.mean((y * unvoiced) ** 2) + 1e-12))
    voiced_body = float(np.sqrt(np.mean((y * voiced) ** 2) + 1e-12))
    target_noise = max(
        unvoiced_body * float(np.mean(unvoiced_amp)),
        voiced_body * float(np.mean(voiced_amp[2:])) * (1.0 - harmonic_mix) * 0.56,
    )
    if noise_rms > 1e-9:
        shaped *= min(7.0, target_noise / noise_rms)
    shaped *= envelope * noise_mask * noise_time

    detail = max(0.0, float(detail_strength))
    detail_gain = 0.45 + 0.55 * detail if detail <= 1.0 else min(1.30, 1.0 + 0.30 * (detail - 1.0))
    branch = (harmonic + shaped) * detail_gain * strength
    branch_rms = float(np.sqrt(np.mean(branch ** 2) + 1e-12))
    safety = 1.0
    safety_limit = (0.10 + 0.16 * strength) * body_rms
    if body_rms > 1e-8 and branch_rms > safety_limit:
        safety = safety_limit / max(branch_rms, 1e-12)
        branch *= safety
        branch_rms *= safety

    return (y.astype(np.float64) + branch).astype(np.float32), {
        "used": True,
        "profile_format": PROFILE_FORMAT,
        "continuity_mode": "source_texture_nonlinear_v2",
        "synthesis_mode": "bandlimited_source_texture",
        "temporal_used": True,
        "harmonic_count": int(harmonic_count),
        "median_f0_hz": float(median_f0),
        "low_pitch_compensation": float(low_pitch_comp),
        "spectral_skirt_mix": float(skirt_mix),
        "spectral_skirt_width_hz": float(skirt_bins * bin_hz),
        "branch_rms": float(branch_rms),
        "safety_gain": float(safety),
        "detail_gain": float(detail_gain),
        "restoration_strength": float(strength),
        "assist_start_hz": float(assist_start_hz),
        "bridge_start_hz": float(bridge_start),
        "voiced_harmonic_mix": float(harmonic_mix),
        "body_edge_ratio_8_12k": float(actual_edge),
        "body_anchor_ratio_6_5_10k": float(lower_anchor),
        "learned_edge_ratio_8_12k": float(learned_edge),
        "reconstruction_floor_ratio": float(anchor_floor),
        "continuity_gain": float(continuity_gain),
        "temporal_upper_peak_gain": float(np.max(upper_t)),
        "temporal_low_peak_gain": float(np.max(low_t)),
        "target_fixed_ms": float(target_fixed_ms),
    }
