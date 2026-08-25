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
        "01_ddsp_raw.wav          24 kHz DDSP compatibility body before Fidelity\n"
        "02_after_fidelity.wav    24 kHz body after Fidelity refiner\n"
        "03_after_articulation.wav 24 kHz body after source-constrained articulation hybrid\n"
        "04_24k_legacy.wav        articulation result resampled to output rate\n"
        "05_fullband.wav          independent fullband DDSP body at output rate\n"
        "06_after_fullband_mix.wav result immediately after legacy/fullband crossover\n"
        "07_final.wav             final signal after highband/topband/loudness processing, before UTAU volume scaling\n"
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
            except Exception:
                _write_stage("07_final.wav", audio, sr)
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
