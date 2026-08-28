#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


CACHE_FORMAT = 1
CACHE_SR = 48000
LOW_HZ = 7600.0
FULL_HZ = 8400.0
TOP_HZ = 20000.0
STATE_DIR_NAME = ".yuaz-0.2.8ai14"
CACHE_DIR_NAME = "source_high_detail"
_INDEX_CACHE = {}


def _mono_float(audio):
    y = np.asarray(audio)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return np.nan_to_num(np.asarray(y, dtype=np.float32).reshape(-1))


def _rms(audio):
    y = np.asarray(audio, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(y * y) + 1e-18)) if y.size else 0.0


def _soft_band(freqs, low_hz=LOW_HZ, full_hz=FULL_HZ, top_hz=TOP_HZ):
    f = np.asarray(freqs, dtype=np.float64)
    out = np.zeros_like(f)
    rise = (f >= low_hz) & (f < full_hz)
    if np.any(rise):
        u = (f[rise] - low_hz) / max(1.0, full_hz - low_hz)
        out[rise] = 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0.0, 1.0))
    out[(f >= full_hz) & (f <= top_hz)] = 1.0
    fall_end = top_hz + 1200.0
    fall = (f > top_hz) & (f < fall_end)
    if np.any(fall):
        u = (f[fall] - top_hz) / 1200.0
        out[fall] = 0.5 + 0.5 * np.cos(np.pi * np.clip(u, 0.0, 1.0))
    return np.clip(out, 0.0, 1.0)


def _frequency_local_median(mag, radius=4):
    mag = np.asarray(mag, dtype=np.float64)
    padded = np.pad(mag, ((radius, radius), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        window_shape=(2 * radius + 1),
        axis=0,
    )
    return np.median(windows, axis=-1)


def extract_source_high_detail(audio, sr, target_sr=CACHE_SR):
    y = _mono_float(audio)
    source_rms = _rms(y)
    if int(sr) != int(target_sr):
        y = librosa.resample(y, orig_sr=int(sr), target_sr=int(target_sr)).astype(np.float32)
    if y.size < 256:
        return np.zeros_like(y, dtype=np.float32), {
            "source_rms": source_rms,
            "detail_rms": 0.0,
            "periodic_scrub_mean": 0.0,
        }

    n_fft = 1024
    hop = 128
    window = np.hanning(n_fft).astype(np.float32)
    spec = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
    )
    mag = np.abs(spec).astype(np.float64)
    freqs = librosa.fft_frequencies(sr=int(target_sr), n_fft=n_fft)
    band = _soft_band(freqs)[:, None]

    # Narrow stable ridges are the part most likely to carry the recorded F0.
    # Scrub them offline while retaining broadband/noisy/transient complex detail.
    local_med = _frequency_local_median(mag, radius=4) + 1e-8
    peak_ratio = mag / local_med
    peakness = np.clip((peak_ratio - 1.55) / 2.80, 0.0, 1.0)
    high_mix = np.clip((freqs - 10500.0) / 4500.0, 0.0, 1.0)[:, None]
    min_keep = 0.34 + 0.20 * high_mix
    periodic_keep = 1.0 - peakness * (1.0 - min_keep)

    # Do not scrub broadband transient frames just because their strongest bins peak.
    frame_flux = np.zeros(mag.shape[1], dtype=np.float64)
    if mag.shape[1] > 1:
        logmag = np.log1p(mag)
        frame_flux[1:] = np.mean(np.maximum(0.0, np.diff(logmag, axis=1)), axis=0)
    if frame_flux.size:
        p75 = float(np.percentile(frame_flux, 75))
        p95 = float(np.percentile(frame_flux, 95))
        transient = np.clip((frame_flux - p75) / max(1e-8, p95 - p75), 0.0, 1.0)[None, :]
        periodic_keep = periodic_keep + transient * (1.0 - periodic_keep) * 0.72

    filtered = spec * band * periodic_keep
    detail = librosa.istft(
        filtered,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        length=len(y),
    ).astype(np.float32)
    detail = np.nan_to_num(detail)
    return detail, {
        "source_rms": float(source_rms),
        "detail_rms": _rms(detail),
        "periodic_scrub_mean": float(np.mean(1.0 - periodic_keep)),
    }


def _cache_file_for(relative_wav):
    digest = hashlib.sha1(str(relative_wav).encode("utf-8", "replace")).hexdigest()[:24]
    return f"{digest}.npy"


def build_voicebank_cache(voicebank_root):
    from .voicebank import scan_voicebank

    root = Path(voicebank_root).expanduser().resolve()
    scan = scan_voicebank(root)
    wavs = {}
    for entry in scan["entries"]:
        path = Path(entry.wav_path).resolve()
        try:
            rel = path.relative_to(root).as_posix()
        except Exception:
            continue
        wavs[rel] = path

    cache_dir = root / STATE_DIR_NAME / CACHE_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    entries = {}
    used = set()
    started = time.perf_counter()

    for i, rel in enumerate(sorted(wavs), 1):
        path = wavs[rel]
        filename = _cache_file_for(rel)
        out = cache_dir / filename
        try:
            raw, sr = sf.read(path, always_2d=False)
            raw = _mono_float(raw)
            stat = path.stat()
            detail, stats = extract_source_high_detail(raw, int(sr), CACHE_SR)
            np.save(out, detail.astype(np.float16), allow_pickle=False)
            used.add(filename)
            entries[rel] = {
                "file": filename,
                "source_sr": int(sr),
                "cache_sr": int(CACHE_SR),
                "source_samples": int(len(raw)),
                "cache_samples": int(len(detail)),
                "source_size": int(stat.st_size),
                "source_mtime_ns": int(stat.st_mtime_ns),
                **stats,
            }
            print(f"[{i}/{len(wavs)}] {rel}", flush=True)
        except Exception as exc:
            print(f"skip {rel}: {exc}", flush=True)

    for path in cache_dir.glob("*.npy"):
        if path.name not in used:
            try:
                path.unlink()
            except Exception:
                pass

    index = {
        "format": CACHE_FORMAT,
        "voicebank_root": str(root),
        "cache_sr": CACHE_SR,
        "low_hz": LOW_HZ,
        "full_hz": FULL_HZ,
        "top_hz": TOP_HZ,
        "extraction": "complex-highband-narrow-ridge-scrub-v1",
        "entries": entries,
        "files": len(entries),
        "build_seconds": float(time.perf_counter() - started),
    }
    tmp = cache_dir / ".index.json.tmp"
    tmp.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, cache_dir / "index.json")
    print(f"Prepared {len(entries)} source high-detail files in {index['build_seconds']:.2f}s")
    print(f"Cache: {cache_dir}")
    return index


def _find_cache(input_path):
    source = Path(input_path).expanduser().resolve()
    for parent in (source.parent, *source.parents):
        index_path = parent / STATE_DIR_NAME / CACHE_DIR_NAME / "index.json"
        if not index_path.is_file():
            continue
        key = str(index_path)
        try:
            mtime = index_path.stat().st_mtime_ns
            cached = _INDEX_CACHE.get(key)
            if cached is None or cached[0] != mtime:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                _INDEX_CACHE[key] = (mtime, data)
            else:
                data = cached[1]
            rel = source.relative_to(parent).as_posix()
            record = (data.get("entries") or {}).get(rel)
            if not record:
                continue
            cache_file = index_path.parent / record["file"]
            if cache_file.is_file():
                return parent, cache_file, record
        except Exception:
            continue
    return None, None, None


def _piecewise_warp_detail(source, source_sr, req, target_samples, output_sr):
    source = np.asarray(source, dtype=np.float32).reshape(-1)
    if source.size < 2 or target_samples < 1:
        return np.zeros(max(1, int(target_samples)), dtype=np.float32)

    source_fixed_ms = max(0.0, float(req.get("consonant", 0.0)))
    velocity = max(1.0, float(req.get("velocity", 100.0)))
    stretch_ratio = 2.0 ** (1.0 - velocity * 0.01)
    target_fixed_ms = source_fixed_ms * stretch_ratio
    target_ms = max(50.0, float(req.get("length", 1000.0)))
    source_total_ms = len(source) * 1000.0 / float(source_sr)

    target_t = np.arange(int(target_samples), dtype=np.float64) * 1000.0 / float(output_sr)
    src_t = np.empty_like(target_t)
    fixed_target = min(target_fixed_ms, target_ms)
    fixed_source = min(source_fixed_ms, source_total_ms)

    fixed_mask = target_t <= fixed_target
    if fixed_target > 1e-6:
        src_t[fixed_mask] = target_t[fixed_mask] * fixed_source / fixed_target
    else:
        src_t[fixed_mask] = 0.0

    tail_target = max(1e-6, target_ms - fixed_target)
    tail_source = max(0.0, source_total_ms - fixed_source)
    src_t[~fixed_mask] = fixed_source + (
        (target_t[~fixed_mask] - fixed_target) * tail_source / tail_target
    )
    src_pos = np.clip(src_t * float(source_sr) / 1000.0, 0.0, len(source) - 1.0)
    return np.interp(src_pos, np.arange(len(source), dtype=np.float64), source).astype(np.float32)


def _highpass_branch(audio, sr):
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if y.size < 32:
        return np.zeros_like(y)
    spec = np.fft.rfft(y.astype(np.float64))
    freqs = np.fft.rfftfreq(len(y), 1.0 / float(sr))
    mask = _soft_band(freqs, LOW_HZ, FULL_HZ, min(TOP_HZ, sr * 0.5 - 50.0))
    return np.fft.irfft(spec * mask, n=len(y)).real.astype(np.float32)


def apply_cached_source_high_detail(req, output_path, strength=0.82):
    root, cache_file, record = _find_cache(req.get("input", ""))
    if cache_file is None:
        return {"used": False, "reason": "no-cache"}

    try:
        from .core import crop_oto

        final, output_sr = sf.read(output_path, always_2d=False)
        final = _mono_float(final)
        if final.size < 32:
            return {"used": False, "reason": "short-output"}

        detail_mm = np.load(cache_file, mmap_mode="r", allow_pickle=False)
        detail = np.asarray(detail_mm, dtype=np.float32)
        cache_sr = int(record.get("cache_sr", CACHE_SR))
        detail = crop_oto(
            detail,
            cache_sr,
            float(req.get("offset", 0.0)),
            float(req.get("cutoff", 0.0)),
        )
        if detail.size < 16:
            return {"used": False, "reason": "short-cache-crop"}

        warped = _piecewise_warp_detail(detail, cache_sr, req, len(final), int(output_sr))
        source_rms = max(float(record.get("source_rms", 0.0)), 1e-8)
        final_rms = max(_rms(final), 1e-8)
        warped *= float(np.clip(final_rms / source_rms, 0.18, 5.0))

        # Preserve the effective brutal-high mechanism, but only with the offline
        # periodicity-scrubbed source branch. Runtime does one FFT, not STFT/DTW.
        final_high = _highpass_branch(final, int(output_sr))
        branch_rms = _rms(warped)
        branch_cap = final_rms * 0.22
        cap_gain = 1.0
        if branch_rms > branch_cap > 1e-9:
            cap_gain = branch_cap / branch_rms
            warped *= cap_gain
            branch_rms *= cap_gain

        mix = float(np.clip(strength, 0.0, 1.0))
        out = final.astype(np.float64) + mix * (
            warped.astype(np.float64) - final_high.astype(np.float64)
        )
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        safety = 1.0
        if peak > 0.985:
            safety = 0.975 / peak
            out *= safety
        out = np.nan_to_num(out).astype(np.float32)

        path = Path(output_path)
        tmp = path.parent / f".{path.name}.source-high-detail-{os.getpid()}-{time.time_ns()}.wav"
        try:
            sf.write(tmp, out, int(output_sr), subtype="PCM_16")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

        return {
            "used": True,
            "backend": "cached-source-complex-high-detail-v1",
            "cache": str(cache_file),
            "strength": mix,
            "branch_rms": float(branch_rms),
            "branch_cap_gain": float(cap_gain),
            "output_safety_gain": float(safety),
        }
    except Exception as exc:
        return {"used": False, "reason": str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("voicebank")
    args = parser.parse_args()
    build_voicebank_cache(args.voicebank)


if __name__ == "__main__":
    main()
