#!/usr/bin/env python3
import argparse
import json
import math
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
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[1] <= 1:
        return np.repeat(matrix[:, :1], len(mapping), axis=1)
    source_x = np.arange(matrix.shape[1], dtype=np.float64)
    mapping = np.clip(mapping, 0.0, matrix.shape[1] - 1)
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


def _build_oracle(source, final, sr, mode, strength):
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
    fin_phase = fin_spec / np.maximum(np.abs(fin_spec), EPS)

    mapping, alignment = _dtw_map(source, final, sr, hop, fin_mag.shape[1])
    if src_mag.shape[1] > 1:
        source_feature_frames = max(1, int(len(source) // hop) + 1)
        mapping *= (src_mag.shape[1] - 1) / max(1, source_feature_frames - 1)
    src_log = _warp_frames(np.log(src_mag), mapping)
    fin_log = np.log(fin_mag)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bin_hz = sr / float(n_fft)

    mid_radius = max(1, int(round(320.0 / bin_hz)))
    high_radius = max(1, int(round(650.0 / bin_hz)))
    mid_src = _moving_average(src_log, mid_radius, axis=0)
    mid_fin = _moving_average(fin_log, mid_radius, axis=0)
    high_src = _moving_average(src_log, high_radius, axis=0)
    high_fin = _moving_average(fin_log, high_radius, axis=0)

    time_radius = max(1, int(round(0.024 * sr / hop)))
    fast_mid_src = mid_src - _moving_average(mid_src, time_radius, axis=1)
    fast_mid_fin = mid_fin - _moving_average(mid_fin, time_radius, axis=1)
    fast_high_src = high_src - _moving_average(high_src, time_radius, axis=1)
    fast_high_fin = high_fin - _moving_average(high_fin, time_radius, axis=1)

    mid_delta = mid_src - mid_fin
    high_delta = high_src - high_fin
    mid_region = (freqs >= 1800.0) & (freqs <= min(8500.0, sr * 0.5 - 100.0))
    high_region = (freqs >= 7500.0) & (freqs <= min(20000.0, sr * 0.5 - 100.0))
    if np.any(mid_region):
        mid_delta -= np.median(mid_delta[mid_region, :])
    if np.any(high_region):
        high_delta -= np.median(high_delta[high_region, :])

    mid_transfer = 0.48 * mid_delta + 0.90 * (fast_mid_src - fast_mid_fin)
    high_transfer = 0.58 * high_delta + 1.00 * (fast_high_src - fast_high_fin)

    nyquist = sr * 0.5
    mid_mask = _raised_band(
        freqs,
        2000.0,
        min(8000.0, nyquist - 300.0),
        500.0,
    )
    high_mask = _raised_band(
        freqs,
        8000.0,
        min(20000.0, nyquist - 250.0),
        750.0,
    )

    if mode == "mid":
        transfer = mid_transfer
        mask = mid_mask
    elif mode == "high":
        transfer = high_transfer
        mask = high_mask
    elif mode == "both":
        mask = np.maximum(mid_mask, high_mask)
        transfer = (
            mid_transfer * mid_mask[:, None]
            + high_transfer * high_mask[:, None]
        ) / np.maximum(mask[:, None], EPS)
    else:
        raise ValueError(mode)

    max_db = 12.0 if mode == "high" else 10.0
    log_limit = max_db * math.log(10.0) / 20.0
    gain_log = np.clip(float(strength) * transfer, -log_limit, log_limit)
    gain_log *= mask[:, None]

    out_mag = fin_mag * np.exp(gain_log)
    out_spec = out_mag * fin_phase
    out = librosa.istft(
        out_spec,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        length=len(final),
    ).astype(np.float32)

    residual = (out - final).astype(np.float32)
    final_rms = _rms(final)
    stats = {
        "mode": mode,
        "alignment": alignment,
        "strength": float(strength),
        "sample_rate": int(sr),
        "n_fft": int(n_fft),
        "hop": int(hop),
        "residual_rms": _rms(residual),
        "final_rms": final_rms,
        "residual_percent": 100.0 * _rms(residual) / max(final_rms, 1e-9),
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
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / "00_final_reference.wav", final, sr, subtype="FLOAT")
    sf.write(out_dir / "00_source_reference.wav", source, sr, subtype="FLOAT")

    report = {
        "source": str(source_path),
        "final": str(final_path),
        "source_original_sr": int(source_sr),
        "analysis_sr": sr,
        "strength": float(args.strength),
        "source_phase_used": False,
        "final_phase_preserved": True,
        "narrow_harmonic_transfer": False,
        "oracles": {},
    }

    for mode in ("mid", "high", "both"):
        out, residual, stats = _build_oracle(
            source,
            final,
            sr,
            mode,
            args.strength,
        )
        sf.write(out_dir / f"oracle_{mode}.wav", out, sr, subtype="FLOAT")
        sf.write(out_dir / f"residual_{mode}.wav", residual, sr, subtype="FLOAT")
        report["oracles"][mode] = stats
        print(
            f"{mode}: residual/final={stats['residual_percent']:.2f}% "
            f"alignment={stats['alignment']}"
        )

    (out_dir / "oracle_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
