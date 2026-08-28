#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .native_detail_oracle import (
    EPS,
    _band_rms,
    _dtw_map,
    _load_mono,
    _raised_band,
    _rms,
    _warp_frames,
)


def _match_level(source, final):
    srms = _rms(source)
    frms = _rms(final)
    if srms > 1e-9 and frms > 1e-9:
        source = source * float(np.clip(frms / srms, 0.20, 5.0))
    return np.asarray(source, dtype=np.float32)


def _build(source, final, sr, mode):
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

    mapping, alignment = _dtw_map(source, final, sr, hop, fin_spec.shape[1])
    if src_spec.shape[1] > 1:
        source_feature_frames = max(1, int(len(source) // hop) + 1)
        mapping *= (src_spec.shape[1] - 1) / max(1, source_feature_frames - 1)
    src_warped = _warp_frames(src_spec, mapping)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    nyquist = sr * 0.5
    if mode == "mid":
        mask = _raised_band(freqs, 1800.0, min(8000.0, nyquist - 250.0), 350.0)
    elif mode == "high":
        mask = _raised_band(freqs, 7600.0, min(20000.0, nyquist - 200.0), 550.0)
    elif mode == "both":
        mask = _raised_band(freqs, 1800.0, min(20000.0, nyquist - 200.0), 400.0)
    elif mode == "all":
        mask = _raised_band(freqs, 450.0, min(20000.0, nyquist - 200.0), 250.0)
    else:
        raise ValueError(mode)

    mask2 = mask[:, None]
    out_spec = fin_spec * (1.0 - mask2) + src_warped * mask2
    out = librosa.istft(
        out_spec,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        length=len(final),
    ).astype(np.float32)

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.15:
        out *= 1.15 / peak

    residual = np.asarray(out - final, dtype=np.float32)
    stats = {
        "mode": mode,
        "alignment": alignment,
        "residual_percent": 100.0 * _rms(residual) / max(_rms(final), 1e-9),
        "bands": {},
    }
    for lo, hi, label in (
        (0, 2000, "0-2k"),
        (2000, 4000, "2-4k"),
        (4000, 8000, "4-8k"),
        (8000, 12000, "8-12k"),
        (12000, 20000, "12-20k"),
    ):
        base = _band_rms(final, sr, lo, hi)
        value = _band_rms(out, sr, lo, hi)
        stats["bands"][label] = {
            "final_rms": base,
            "oracle_rms": value,
            "energy_ratio": value / max(base, EPS),
        }
    return out, residual, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dump_dir",
        nargs="?",
        default=str(Path.home() / "Desktop" / "YuazClarityDump"),
    )
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
    source = _match_level(source, final)
    sr = int(final_sr)

    out_dir = dump / "brutal_oracle"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / "00_final_reference.wav", final, sr, subtype="FLOAT")
    sf.write(out_dir / "00_source_reference.wav", source, sr, subtype="FLOAT")

    report = {
        "format": 1,
        "purpose": "diagnostic upper bound only",
        "source_phase_used": True,
        "source_periodicity_can_leak": True,
        "not_for_runtime": True,
        "oracles": {},
    }

    for mode in ("mid", "high", "both", "all"):
        out, residual, stats = _build(source, final, sr, mode)
        sf.write(out_dir / f"brutal_{mode}.wav", out, sr, subtype="FLOAT")
        sf.write(out_dir / f"residual_{mode}.wav", residual, sr, subtype="FLOAT")
        report["oracles"][mode] = stats
        print(
            f"{mode}: residual/final={stats['residual_percent']:.2f}% "
            f"2-4k={stats['bands']['2-4k']['energy_ratio']:.2f}x "
            f"4-8k={stats['bands']['4-8k']['energy_ratio']:.2f}x "
            f"8-12k={stats['bands']['8-12k']['energy_ratio']:.2f}x "
            f"12-20k={stats['bands']['12-20k']['energy_ratio']:.2f}x"
        )

    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
