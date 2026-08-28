#!/usr/bin/env python3
import argparse
import json
import math
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


EPS = 1e-8


def _load_mono(path):
    y, sr = sf.read(path, always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = np.mean(y, axis=1)
    y = np.nan_to_num(np.asarray(y, dtype=np.float32).reshape(-1))
    return y, int(sr)


def _rms(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if not x.size:
        return 0.0
    return float(np.sqrt(np.mean(x * x) + 1e-18))


def _moving_average(x, radius, axis):
    radius = int(max(0, radius))
    if radius <= 0:
        return np.asarray(x, dtype=np.float64).copy()
    kernel = np.ones(2 * radius + 1, dtype=np.float64) / float(2 * radius + 1)
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"),
        axis,
        np.asarray(x, dtype=np.float64),
    )


def _raised_band(freqs, lo, hi, transition):
    freqs = np.asarray(freqs, dtype=np.float64)
    lo = float(lo)
    hi = float(hi)
    transition = max(1.0, float(transition))
    mask = np.zeros_like(freqs)
    mask[(freqs >= lo) & (freqs <= hi)] = 1.0

    left = (freqs >= max(0.0, lo - transition)) & (freqs < lo)
    if np.any(left):
        u = (freqs[left] - (lo - transition)) / transition
        mask[left] = 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0.0, 1.0))

    right = (freqs > hi) & (freqs <= hi + transition)
    if np.any(right):
        u = (freqs[right] - hi) / transition
        mask[right] = 0.5 + 0.5 * np.cos(np.pi * np.clip(u, 0.0, 1.0))
    return np.clip(mask, 0.0, 1.0)


def _alignment_features(y, sr, hop):
    n_fft = 1024 if sr >= 32000 else 512
    fmax = min(7600.0, sr * 0.5 - 100.0)
    feat = librosa.feature.melspectrogram(
        y=np.asarray(y, dtype=np.float32),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        n_mels=24,
        fmin=250.0,
        fmax=fmax,
        power=1.0,
        center=True,
    ).astype(np.float64)
    feat = np.log1p(12.0 * feat)
    med = np.median(feat, axis=1, keepdims=True)
    scale = np.std(feat, axis=1, keepdims=True) + 1e-4
    return np.clip((feat - med) / scale, -5.0, 5.0)


def _linear_map(src_frames, dst_frames):
    if dst_frames <= 1:
        return np.zeros(max(1, dst_frames), dtype=np.float64)
    return np.linspace(0.0, max(0, src_frames - 1), dst_frames, dtype=np.float64)


def _dtw_map(source, final, sr, hop, dst_frames):
    try:
        src_feat = _alignment_features(source, sr, hop)
        dst_feat = _alignment_features(final, sr, hop)
        _, path = librosa.sequence.dtw(
            X=src_feat,
            Y=dst_feat,
            metric="cosine",
            backtrack=True,
        )
        buckets = [[] for _ in range(dst_feat.shape[1])]
        for src_i, dst_i in np.asarray(path, dtype=np.int64):
            if 0 <= dst_i < len(buckets):
                buckets[int(dst_i)].append(int(src_i))

        mapping = np.full(dst_feat.shape[1], np.nan, dtype=np.float64)
        for i, values in enumerate(buckets):
            if values:
                mapping[i] = float(np.median(values))
        known = np.flatnonzero(np.isfinite(mapping))
        if known.size < 2:
            raise RuntimeError("DTW map is too sparse")
        mapping = np.interp(np.arange(mapping.size), known, mapping[known])
        mapping = np.maximum.accumulate(mapping)
        mapping = np.clip(mapping, 0.0, max(0, src_feat.shape[1] - 1))
        if mapping.size != dst_frames:
            x0 = np.linspace(0.0, 1.0, mapping.size)
            x1 = np.linspace(0.0, 1.0, dst_frames)
            mapping = np.interp(x1, x0, mapping)
        return mapping, "dtw"
    except Exception:
        src_frames = max(1, int(len(source) // hop) + 1)
        return _linear_map(src_frames, dst_frames), "linear"


def _warp_frames(matrix, mapping):
    matrix = np.asarray(matrix)
    if matrix.shape[1] <= 1:
        return np.repeat(matrix[:, :1], len(mapping), axis=1)
    source_x = np.arange(matrix.shape[1], dtype=np.float64)
    mapping = np.clip(mapping, 0.0, matrix.shape[1] - 1)
    if np.iscomplexobj(matrix):
        real = np.vstack([np.interp(mapping, source_x, row.real) for row in matrix])
        imag = np.vstack([np.interp(mapping, source_x, row.imag) for row in matrix])
        return real + 1j * imag
    return np.vstack([np.interp(mapping, source_x, row) for row in matrix])


def _band_rms(y, sr, lo, hi):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size < 32:
        return 0.0
    n = int(1 << int(np.ceil(np.log2(max(32, y.size)))))
    spec = np.fft.rfft(y, n=n)
    freqs = np.fft.rfftfreq(n, 1.0 / float(sr))
    mask = (freqs >= float(lo)) & (freqs < min(float(hi), sr * 0.5))
    if not np.any(mask):
        return 0.0
    return float(
        np.sqrt(np.mean(np.abs(spec[mask]) ** 2) + 1e-18)
        / max(1.0, np.sqrt(n))
    )


def _match_source_level(src_log, fin_log, freqs):
    anchor = (freqs >= 900.0) & (freqs <= min(3600.0, freqs[-1] - 100.0))
    if not np.any(anchor):
        return src_log
    offset = np.median(fin_log[anchor, :] - src_log[anchor, :], axis=0)
    offset = _moving_average(offset.reshape(1, -1), 3, axis=1).reshape(-1)
    offset = np.clip(offset, -2.3, 2.3)
    return src_log + offset[None, :]


def _hybrid_phase(fin_spec, fin_mag, seed):
    rng = np.random.default_rng(int(seed))
    random_phase = np.exp(1j * rng.uniform(-np.pi, np.pi, size=fin_spec.shape))
    final_phase = fin_spec / np.maximum(fin_mag, EPS)

    frame_peak = np.max(fin_mag, axis=0, keepdims=True)
    threshold = np.maximum(frame_peak * 2.5e-3, 2.5e-7)
    present = np.clip((fin_mag - threshold) / np.maximum(3.0 * threshold, EPS), 0.0, 1.0)
    phase = present * final_phase + (1.0 - present) * random_phase
    phase /= np.maximum(np.abs(phase), EPS)
    return phase, present


def _source_targets(src_log_matched, freqs, sr, n_fft):
    bin_hz = sr / float(n_fft)
    safe_mid_radius = max(1, int(round(220.0 / bin_hz)))
    safe_high_radius = max(1, int(round(480.0 / bin_hz)))

    safe_mid_log = _moving_average(src_log_matched, safe_mid_radius, axis=0)
    safe_high_log = _moving_average(src_log_matched, safe_high_radius, axis=0)

    time_radius = 3
    safe_mid_log += 0.55 * (
        safe_mid_log - _moving_average(safe_mid_log, time_radius, axis=1)
    )
    safe_high_log += 0.70 * (
        safe_high_log - _moving_average(safe_high_log, time_radius, axis=1)
    )

    upper_radius = max(1, int(round(55.0 / bin_hz)))
    upper_log = _moving_average(src_log_matched, upper_radius, axis=0)

    return {
        "safe_mid": np.exp(np.clip(safe_mid_log, -18.0, 6.0)),
        "safe_high": np.exp(np.clip(safe_high_log, -18.0, 6.0)),
        "upper": np.exp(np.clip(upper_log, -18.0, 6.0)),
    }


def _limit_residual(final, out, limit_ratio):
    residual = np.asarray(out - final, dtype=np.float32)
    fr = _rms(final)
    rr = _rms(residual)
    limit = float(limit_ratio) * fr
    if rr > limit > 1e-9:
        residual *= float(limit / rr)
        out = final + residual
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 1.15:
        scale = 1.15 / peak
        out = out * scale
        residual = out - final
    return np.asarray(out, dtype=np.float32), np.asarray(residual, dtype=np.float32)


def _build_oracle(source, final, sr, mode, style, strength):
    n_fft = 2048 if sr >= 32000 else 1024
    hop = max(64, n_fft // 16)
    window = np.hanning(n_fft).astype(np.float32)

    src_spec = librosa.stft(
        source,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
    )
    fin_spec = librosa.stft(
        final,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
    )
    src_mag = np.maximum(np.abs(src_spec).astype(np.float64), EPS)
    fin_mag = np.maximum(np.abs(fin_spec).astype(np.float64), EPS)

    mapping, alignment = _dtw_map(source, final, sr, hop, fin_mag.shape[1])
    if src_mag.shape[1] > 1:
        source_feature_frames = max(1, int(len(source) // hop) + 1)
        mapping *= (src_mag.shape[1] - 1) / max(1, source_feature_frames - 1)
    src_log = _warp_frames(np.log(src_mag), mapping)
    fin_log = np.log(fin_mag)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    src_log = _match_source_level(src_log, fin_log, freqs)
    targets = _source_targets(src_log, freqs, sr, n_fft)

    nyquist = sr * 0.5
    mid_mask = _raised_band(
        freqs,
        1800.0,
        min(8000.0, nyquist - 300.0),
        550.0,
    )
    high_mask = _raised_band(
        freqs,
        7600.0,
        min(20000.0, nyquist - 250.0),
        800.0,
    )

    if mode == "mid":
        mask = mid_mask
        src_target = targets["safe_mid"] if style == "safe" else targets["upper"]
    elif mode == "high":
        mask = high_mask
        src_target = targets["safe_high"] if style == "safe" else targets["upper"]
    elif mode == "both":
        mask = np.maximum(mid_mask, high_mask)
        if style == "safe":
            src_target = (
                targets["safe_mid"] * mid_mask[:, None]
                + targets["safe_high"] * high_mask[:, None]
            ) / np.maximum(mask[:, None], EPS)
        else:
            src_target = targets["upper"]
    else:
        raise ValueError(mode)

    # v2 is deliberately additive/reconstructive rather than multiplicative.
    # Missing final bins can now receive real source-derived magnitude.
    desired = np.maximum(fin_mag, src_target)
    if style == "safe":
        desired = np.minimum(desired, np.maximum(fin_mag * 7.0, np.median(src_target, axis=0, keepdims=True) * 5.0))
    else:
        desired = np.minimum(desired, np.maximum(fin_mag * 14.0, np.median(src_target, axis=0, keepdims=True) * 9.0))

    blend = np.clip(float(strength) * mask[:, None], 0.0, 1.0)
    out_mag = fin_mag * (1.0 - blend) + desired * blend

    seed = 20260828 + (0 if style == "safe" else 1000) + {"mid": 1, "high": 2, "both": 3}[mode]
    phase, presence = _hybrid_phase(fin_spec, fin_mag, seed)
    out_spec = out_mag * phase
    out = librosa.istft(
        out_spec,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        length=len(final),
    ).astype(np.float32)

    out, residual = _limit_residual(
        final,
        out,
        0.30 if style == "safe" else 0.55,
    )

    final_rms = _rms(final)
    stats = {
        "style": style,
        "mode": mode,
        "alignment": alignment,
        "strength": float(strength),
        "sample_rate": int(sr),
        "n_fft": int(n_fft),
        "hop": int(hop),
        "residual_rms": _rms(residual),
        "final_rms": final_rms,
        "residual_percent": 100.0 * _rms(residual) / max(final_rms, 1e-9),
        "random_phase_fraction": float(np.mean(1.0 - presence)),
        "bands": {},
    }
    for lo, hi, label in (
        (2000, 4000, "2-4k"),
        (4000, 8000, "4-8k"),
        (8000, 12000, "8-12k"),
        (12000, 20000, "12-20k"),
    ):
        base = _band_rms(final, sr, lo, hi)
        oracle = _band_rms(out, sr, lo, hi)
        stats["bands"][label] = {
            "final_rms": base,
            "oracle_rms": oracle,
            "energy_ratio": oracle / max(base, 1e-12),
        }
    return out, residual, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dump_dir",
        nargs="?",
        default=str(Path.home() / "Desktop" / "YuazClarityDump"),
    )
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()

    dump = Path(args.dump_dir).expanduser().resolve()
    source_path = dump / "00_source_native.wav"
    final_path = dump / "07_final.wav"
    if not source_path.exists():
        raise RuntimeError(f"Missing {source_path}")
    if not final_path.exists():
        raise RuntimeError(f"Missing {final_path}")

    source, source_sr = _load_mono(source_path)
    final, final_sr = _load_mono(final_path)
    if source_sr != final_sr:
        source = librosa.resample(
            source,
            orig_sr=source_sr,
            target_sr=final_sr,
        ).astype(np.float32)

    sr = int(final_sr)
    source_rms = _rms(source)
    final_rms = _rms(final)
    if source_rms > 1e-8 and final_rms > 1e-8:
        source *= float(np.clip(final_rms / source_rms, 0.25, 4.0))

    out_dir = dump / "oracle"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / "00_final_reference.wav", final, sr, subtype="FLOAT")
    sf.write(out_dir / "00_source_reference.wav", source, sr, subtype="FLOAT")

    report = {
        "format": 2,
        "source": str(source_path),
        "final": str(final_path),
        "source_original_sr": int(source_sr),
        "analysis_sr": sr,
        "strength": float(args.strength),
        "source_phase_used": False,
        "missing_bins_can_be_created": True,
        "safe": "frequency-smoothed source envelope plus temporal detail; hybrid final/random phase",
        "upper": "lightly smoothed source magnitude upper-bound diagnostic; hybrid final/random phase",
        "oracles": {},
    }

    for style in ("safe", "upper"):
        for mode in ("mid", "high", "both"):
            out, residual, stats = _build_oracle(
                source,
                final,
                sr,
                mode,
                style,
                args.strength,
            )
            key = f"{style}_{mode}"
            sf.write(out_dir / f"oracle_{key}.wav", out, sr, subtype="FLOAT")
            sf.write(out_dir / f"residual_{key}.wav", residual, sr, subtype="FLOAT")
            report["oracles"][key] = stats
            print(
                f"{key}: residual/final={stats['residual_percent']:.2f}% "
                f"alignment={stats['alignment']} "
                f"8-12k={stats['bands']['8-12k']['energy_ratio']:.2f}x "
                f"12-20k={stats['bands']['12-20k']['energy_ratio']:.2f}x"
            )

    (out_dir / "oracle_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
