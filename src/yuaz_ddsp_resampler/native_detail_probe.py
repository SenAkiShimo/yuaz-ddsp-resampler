#!/usr/bin/env python3
import threading
from pathlib import Path

import numpy as np
import soundfile as sf


_state = threading.local()
BANDS = (
    (0, 2000, "0-2k"),
    (2000, 4000, "2-4k"),
    (4000, 8000, "4-8k"),
    (8000, 12000, "8-12k"),
    (12000, 20000, "12-20k"),
)


def _mono_float(audio):
    y = np.asarray(audio)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = np.nan_to_num(np.asarray(y, dtype=np.float32).reshape(-1))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def capture_native_read(path):
    try:
        raw, sr = sf.read(path, always_2d=False)
        _state.native = (str(path), _mono_float(raw), int(sr))
    except Exception:
        _state.native = None


def capture_crop(cropped, sr, offset_ms, cutoff_ms, crop_fn):
    try:
        _state.source24 = (_mono_float(cropped), int(sr))
        native = getattr(_state, "native", None)
        if native is None:
            _state.native_crop = None
            return
        _, raw, raw_sr = native
        native_crop = crop_fn(raw, raw_sr, offset_ms, cutoff_ms)
        _state.native_crop = (_mono_float(native_crop), int(raw_sr))
    except Exception:
        _state.source24 = None
        _state.native_crop = None


def write_pending_sources(dump_dir):
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("00_source_native.wav", getattr(_state, "native_crop", None)),
        ("00_source_24k.wav", getattr(_state, "source24", None)),
    ):
        if value is None:
            continue
        try:
            audio, sr = value
            sf.write(dump_dir / name, audio, int(sr), subtype="FLOAT")
        except Exception:
            pass


def _rms(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(x * x) + 1e-18)) if x.size else 0.0


def _band_rms(x, sr, lo, hi):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 32:
        return 0.0
    n = int(1 << int(np.ceil(np.log2(max(32, x.size)))))
    spec = np.fft.rfft(x, n=n)
    freqs = np.fft.rfftfreq(n, 1.0 / float(sr))
    upper = min(float(hi), float(sr) * 0.5)
    mask = (freqs >= float(lo)) & (freqs < upper)
    if not np.any(mask):
        return 0.0
    power = np.abs(spec[mask]) ** 2
    return float(np.sqrt(np.mean(power) + 1e-18) / max(1.0, np.sqrt(n)))


def _band_signal(x, sr, lo, hi):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 32:
        return np.zeros_like(x, dtype=np.float32)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / float(sr))
    upper = min(float(hi), float(sr) * 0.5)
    mask = (freqs >= float(lo)) & (freqs < upper)
    spec[~mask] = 0.0
    return np.fft.irfft(spec, n=x.size).real.astype(np.float32)


def _detail_metrics(x, sr, lo, hi):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    result = {
        "rms": _band_rms(x, sr, lo, hi),
        "flux_mean": 0.0,
        "flux_p95": 0.0,
        "env_delta": 0.0,
        "env_crest": 0.0,
    }
    if x.size < 128:
        return result

    n_fft = 512 if int(sr) <= 24000 else 1024
    max_fft = 1 << int(np.floor(np.log2(max(128, x.size))))
    n_fft = min(n_fft, max_fft)
    hop = max(32, n_fft // 8)
    window = np.hanning(n_fft)
    rows = []
    starts = list(range(0, max(1, x.size - n_fft + 1), hop))
    if not starts:
        starts = [0]
    for start in starts:
        frame = x[start:start + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size))
        mag = np.abs(np.fft.rfft(frame * window))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sr))
        mask = (freqs >= float(lo)) & (freqs < min(float(hi), float(sr) * 0.5))
        if np.any(mask):
            rows.append(mag[mask])
    if not rows:
        return result

    mag = np.stack(rows, axis=1)
    env = np.sqrt(np.mean(mag * mag, axis=0) + 1e-18)
    logmag = np.log1p(mag)
    if logmag.shape[1] > 1:
        flux = np.mean(np.maximum(0.0, np.diff(logmag, axis=1)), axis=0)
        env_d = np.diff(np.log(env + 1e-12))
        result["flux_mean"] = float(np.mean(flux))
        result["flux_p95"] = float(np.percentile(flux, 95))
        result["env_delta"] = float(np.sqrt(np.mean(env_d * env_d) + 1e-18))
    result["env_crest"] = float(np.percentile(env, 95) / max(np.median(env), 1e-12))
    return result


def _load(path):
    y, sr = sf.read(path, always_2d=False)
    return _mono_float(y).astype(np.float64), int(sr)


def write_detail_report(dump_dir):
    dump_dir = Path(dump_dir)
    source_path = dump_dir / "00_source_native.wav"
    final_path = dump_dir / "07_final.wav"
    if not source_path.exists() or not final_path.exists():
        return

    names = (
        "00_source_native.wav",
        "00_source_24k.wav",
        "01_ddsp_raw.wav",
        "03_after_articulation.wav",
        "05_fullband.wav",
        "06_after_fullband_mix.wav",
        "07_final.wav",
    )
    loaded = {}
    for name in names:
        path = dump_dir / name
        if not path.exists():
            continue
        try:
            loaded[name] = _load(path)
        except Exception:
            pass
    if "00_source_native.wav" not in loaded:
        return

    metrics = {}
    lines = [
        "Yuaz native source-detail diagnostic",
        "Statistics are compared independently of duration and pitch alignment.",
        "rms = average energy in the band",
        "flux = positive short-time spectral change; lower than source can indicate smeared transients/detail",
        "env_delta = frame-to-frame log band-envelope motion; lower than source can indicate temporal smoothing",
        "env_crest = p95 / median band-envelope contrast",
        "",
    ]

    for name in names:
        if name not in loaded:
            lines.append(f"{name}: missing")
            continue
        y, sr = loaded[name]
        metrics[name] = {}
        lines.append(f"{name}: sr={sr} duration={len(y) / max(sr, 1):.4f}s total_rms={_rms(y):.8f}")
        for lo, hi, label in BANDS:
            m = _detail_metrics(y, sr, lo, hi)
            metrics[name][label] = m
            lines.append(
                f"  {label}: rms={m['rms']:.8f} flux_mean={m['flux_mean']:.6f} "
                f"flux_p95={m['flux_p95']:.6f} env_delta={m['env_delta']:.6f} "
                f"env_crest={m['env_crest']:.3f}"
            )
        lines.append("")

    source = metrics["00_source_native.wav"]
    lines += ["SOURCE-NORMALIZED RATIOS (stage / native source)", ""]
    for name in names[1:]:
        if name not in metrics:
            continue
        lines.append(name)
        for _, _, label in BANDS:
            sm = source[label]
            mm = metrics[name][label]
            lines.append(
                f"  {label}: energy={mm['rms'] / max(sm['rms'], 1e-12):.3f}x "
                f"flux={mm['flux_mean'] / max(sm['flux_mean'], 1e-12):.3f}x "
                f"env_delta={mm['env_delta'] / max(sm['env_delta'], 1e-12):.3f}x"
            )
        lines.append("")

    try:
        (dump_dir / "09_detail_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass

    try:
        band_dir = dump_dir / "bands"
        band_dir.mkdir(parents=True, exist_ok=True)
        for name, stem in (("00_source_native.wav", "source"), ("07_final.wav", "final")):
            if name not in loaded:
                continue
            y, sr = loaded[name]
            for lo, hi, label in BANDS:
                sf.write(
                    band_dir / f"{stem}_{label}.wav",
                    _band_signal(y, sr, lo, hi),
                    sr,
                    subtype="FLOAT",
                )
    except Exception:
        pass
