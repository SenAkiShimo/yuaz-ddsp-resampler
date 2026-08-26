#!/usr/bin/env python3
import shutil
import threading
from pathlib import Path

import numpy as np
import soundfile as sf


_state = threading.local()
_installed = False
_original_decode_dualrate = None
_original_articulation = None
_original_blends = {}
_original_write_wav = None


def set_mode(value):
    global _installed
    _state.mode = float(np.clip(float(value), 0.0, 100.0))
    if not _installed:
        _install()


def get_mode():
    return float(getattr(_state, "mode", 0.0))


def _active():
    return get_mode() >= 99.0


def _dump_dir():
    desktop = Path.home() / "Desktop"
    root = desktop if desktop.exists() else Path.home()
    return root / "YuazClarityDump"


def _reset_dump():
    if not _active():
        return
    path = _dump_dir()
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    (path / "00_README.txt").write_text(
        "Yuaz clarity stage dump\n"
        "\n"
        "01_ddsp_raw.wav            24 kHz DDSP compatibility body before Fidelity\n"
        "02_after_fidelity.wav      24 kHz body after Fidelity refiner\n"
        "03_after_articulation.wav  24 kHz body after source-constrained articulation hybrid\n"
        "04_24k_legacy.wav          articulation result resampled to output rate\n"
        "05_fullband.wav            independent fullband DDSP body at output rate\n"
        "06_after_fullband_mix.wav  result immediately after legacy/fullband crossover\n"
        "07_final.wav               final signal after highband/topband/loudness processing, before UTAU volume scaling\n"
        "08_comparison.txt          objective numeric comparison between stages\n"
        "\n"
        "YC100 enables dump only. It does not alter synthesis.\n",
        encoding="utf-8",
    )


def _write_stage(name, audio, sr):
    if not _active():
        return
    try:
        path = _dump_dir()
        path.mkdir(parents=True, exist_ok=True)
        y = np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1))
        sf.write(path / name, y, int(sr), subtype="FLOAT")
    except Exception:
        pass


def _linear_resample(x, orig_sr, target_sr):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 2 or int(orig_sr) == int(target_sr):
        return x.copy()
    n = max(1, int(round(x.size * float(target_sr) / float(orig_sr))))
    src = np.linspace(0.0, 1.0, x.size, endpoint=True)
    dst = np.linspace(0.0, 1.0, n, endpoint=True)
    return np.interp(dst, src, x)


def _rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x) + 1e-18)) if x.size else 0.0


def _band_rms(x, sr, lo, hi):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 32:
        return 0.0
    n = int(1 << int(np.ceil(np.log2(max(32, x.size)))))
    spec = np.fft.rfft(x, n=n)
    freqs = np.fft.rfftfreq(n, 1.0 / float(sr))
    mask = (freqs >= float(lo)) & (freqs < min(float(hi), float(sr) * 0.5))
    if not np.any(mask):
        return 0.0
    power = np.abs(spec[mask]) ** 2
    return float(np.sqrt(np.mean(power) + 1e-18) / max(1.0, np.sqrt(n)))


def _pair_metrics(a, b):
    n = min(a.size, b.size)
    if n < 2:
        return {"corr": 0.0, "diff_rms": 0.0, "diff_db_rel": -120.0, "max_abs": 0.0}
    a = a[:n]
    b = b[:n]
    d = b - a
    ar = _rms(a)
    dr = _rms(d)
    ac = a - np.mean(a)
    bc = b - np.mean(b)
    denom = float(np.sqrt(np.sum(ac * ac) * np.sum(bc * bc)) + 1e-18)
    corr = float(np.sum(ac * bc) / denom) if denom > 0.0 else 0.0
    rel = 20.0 * np.log10(max(dr, 1e-12) / max(ar, 1e-12))
    return {
        "corr": corr,
        "diff_rms": dr,
        "diff_db_rel": float(rel),
        "max_abs": float(np.max(np.abs(d))) if d.size else 0.0,
    }


def _write_comparison_report():
    if not _active():
        return
    path = _dump_dir()
    names = [
        "01_ddsp_raw.wav",
        "02_after_fidelity.wav",
        "03_after_articulation.wav",
        "04_24k_legacy.wav",
        "05_fullband.wav",
        "06_after_fullband_mix.wav",
        "07_final.wav",
    ]
    loaded = {}
    target_sr = 48000
    for name in names:
        file = path / name
        if not file.exists():
            continue
        try:
            y, sr = sf.read(file, always_2d=False)
            if getattr(y, "ndim", 1) > 1:
                y = np.mean(y, axis=1)
            y = np.nan_to_num(np.asarray(y, dtype=np.float64).reshape(-1))
            loaded[name] = (_linear_resample(y, int(sr), target_sr), int(sr))
        except Exception:
            continue

    lines = [
        "Yuaz clarity stage comparison",
        "all comparison waveforms resampled to 48000 Hz with linear interpolation",
        "diff_db_rel = RMS(stage difference) relative to RMS(reference); more negative means more similar",
        "",
        "STAGE LEVELS",
    ]
    bands = [(0, 2000), (2000, 4000), (4000, 8000), (8000, 12000), (12000, 20000)]
    for name in names:
        if name not in loaded:
            lines.append(f"{name}: missing")
            continue
        y, original_sr = loaded[name]
        band_text = " ".join(
            f"{lo//1000}-{hi//1000}k={_band_rms(y, target_sr, lo, hi):.7f}" for lo, hi in bands
        )
        lines.append(
            f"{name}: original_sr={original_sr} samples48={y.size} rms={_rms(y):.8f} {band_text}"
        )

    lines += ["", "ADJACENT STAGE DIFFERENCES"]
    for a_name, b_name in zip(names[:-1], names[1:]):
        if a_name not in loaded or b_name not in loaded:
            continue
        m = _pair_metrics(loaded[a_name][0], loaded[b_name][0])
        lines.append(
            f"{a_name} -> {b_name}: corr={m['corr']:.8f} diff_rms={m['diff_rms']:.8f} "
            f"diff_db_rel={m['diff_db_rel']:.2f}dB max_abs={m['max_abs']:.8f}"
        )

    if names[0] in loaded:
        ref = loaded[names[0]][0]
        lines += ["", "DIFFERENCE FROM 01_ddsp_raw.wav"]
        for name in names[1:]:
            if name not in loaded:
                continue
            m = _pair_metrics(ref, loaded[name][0])
            lines.append(
                f"01 -> {name}: corr={m['corr']:.8f} diff_rms={m['diff_rms']:.8f} "
                f"diff_db_rel={m['diff_db_rel']:.2f}dB max_abs={m['max_abs']:.8f}"
            )

    try:
        (path / "08_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass


def _install():
    global _installed, _original_decode_dualrate, _original_articulation, _original_write_wav
    if _installed:
        return

    from . import core

    _original_decode_dualrate = core.deterministic_decode_dualrate
    _original_articulation = core.articulation_hybrid_mix
    _original_write_wav = core.write_wav

    def decode_dualrate_wrapper(*args, **kwargs):
        legacy, fullband, stats = _original_decode_dualrate(*args, **kwargs)
        if _active():
            _reset_dump()
            decoder = args[0] if args else kwargs.get("decoder")
            analysis_sr = int(getattr(decoder, "sample_rate", 24000))
            _write_stage("01_ddsp_raw.wav", legacy, analysis_sr)
        return legacy, fullband, stats

    def articulation_wrapper(
        original, generated, sr, source_f0, target_f0, regions,
        source_fixed_ms, target_fixed_ms, target_ms, canonical_template=None,
    ):
        if _active():
            _write_stage("02_after_fidelity.wav", generated, sr)
        mixed, stats = _original_articulation(
            original, generated, sr, source_f0, target_f0, regions,
            source_fixed_ms, target_fixed_ms, target_ms,
            canonical_template=canonical_template,
        )
        if _active():
            _write_stage("03_after_articulation.wav", mixed, sr)
        return mixed, stats

    def make_blend_wrapper(name):
        original = getattr(core, name)
        _original_blends[name] = original

        def wrapper(legacy_output, fullband_output, sr, *args, **kwargs):
            if _active():
                _write_stage("04_24k_legacy.wav", legacy_output, sr)
                _write_stage("05_fullband.wav", fullband_output, sr)
            mixed, stats = original(legacy_output, fullband_output, sr, *args, **kwargs)
            if _active():
                _write_stage("06_after_fullband_mix.wav", mixed, sr)
            return mixed, stats

        return wrapper

    def write_wav_wrapper(path, audio, sr, volume=100.0):
        if _active():
            try:
                target = Path(path).expanduser().resolve()
                dump = _dump_dir().expanduser().resolve()
                if dump not in target.parents:
                    _write_stage("07_final.wav", audio, sr)
                    _write_comparison_report()
            except Exception:
                _write_stage("07_final.wav", audio, sr)
                _write_comparison_report()
        return _original_write_wav(path, audio, sr, volume)

    core.deterministic_decode_dualrate = decode_dualrate_wrapper
    core.articulation_hybrid_mix = articulation_wrapper
    for name in (
        "blend_dualrate_fullband_body",
        "blend_dualrate_fullband_body_v2",
        "blend_dualrate_fullband_body_v3",
    ):
        if hasattr(core, name):
            setattr(core, name, make_blend_wrapper(name))
    core.write_wav = write_wav_wrapper
    _installed = True
