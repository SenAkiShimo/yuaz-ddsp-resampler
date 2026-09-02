#!/usr/bin/env python3
import os
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from . import core
from .high_detail_tf_runtime import _exact_f0_tracks
from .voicebank import parse_oto_file


N_FFT = 2048
HOP = 128
LOW_HZ = 7200.0
FULL_HZ = 7800.0
TOP_HZ = 20000.0
_OTO_CACHE = {}


def _mono(x):
    y = np.asarray(x)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return np.nan_to_num(np.asarray(y, dtype=np.float32).reshape(-1))


def _rms(x):
    y = np.asarray(x, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(y * y) + 1e-12)) if y.size else 0.0


def _voicebank_root(input_path):
    p = Path(input_path).expanduser().resolve()
    for parent in (p.parent, *p.parents):
        if (parent / ".yuaz-0.2.8ai14").is_dir():
            return parent
        if (parent / "character.txt").is_file() and any(parent.rglob("oto.ini")):
            return parent
    return p.parent


def _local_oto_entry(req):
    wav = Path(req.get("input", "")).expanduser().resolve()
    root = _voicebank_root(wav)
    candidates = []
    for parent in (wav.parent, *wav.parents[:2]):
        oto = parent / "oto.ini"
        if oto.is_file():
            candidates.append(oto)
    best = None
    best_score = float("inf")
    for oto in candidates:
        try:
            key = str(oto.resolve())
            mtime = oto.stat().st_mtime_ns
            cached = _OTO_CACHE.get(key)
            if cached is None or cached[0] != mtime:
                entries, _, _ = parse_oto_file(oto, root)
                _OTO_CACHE[key] = (mtime, entries)
            else:
                entries = cached[1]
            for entry in entries:
                if Path(entry.wav_path).resolve() != wav:
                    continue
                score = (
                    abs(float(entry.offset) - float(req.get("offset", 0.0)))
                    + 0.35 * abs(float(entry.cutoff) - float(req.get("cutoff", 0.0)))
                    + 0.20 * abs(float(entry.consonant) - float(req.get("consonant", 0.0)))
                )
                if score < best_score:
                    best_score = score
                    best = entry
        except Exception:
            continue
    return best


def _landmark_warp(source, source_sr, req, target_samples, target_sr, oto_entry=None):
    x = np.asarray(source, dtype=np.float32).reshape(-1)
    if x.size < 2 or int(target_samples) < 2:
        return np.zeros(max(1, int(target_samples)), dtype=np.float32), {}

    velocity = max(1.0, float(req.get("velocity", 100.0)))
    stretch = 2.0 ** (1.0 - velocity * 0.01)
    s_con = max(0.0, float(req.get("consonant", 0.0)))
    s_pre = max(0.0, min(s_con, float(getattr(oto_entry, "preutterance", 0.0) or 0.0)))
    t_pre = s_pre * stretch
    t_con = s_con * stretch
    source_total = len(x) * 1000.0 / float(source_sr)
    target_total = int(target_samples) * 1000.0 / float(target_sr)

    s_pts = np.array([0.0, min(s_pre, source_total), min(s_con, source_total), source_total], dtype=np.float64)
    t_pts = np.array([0.0, min(t_pre, target_total), min(t_con, target_total), target_total], dtype=np.float64)
    for pts in (s_pts, t_pts):
        for i in range(1, len(pts)):
            if pts[i] <= pts[i - 1]:
                pts[i] = pts[i - 1] + 1e-3

    target_t = np.arange(int(target_samples), dtype=np.float64) * 1000.0 / float(target_sr)
    source_t = np.interp(target_t, t_pts, s_pts)
    pos = np.clip(source_t * float(source_sr) / 1000.0, 0.0, len(x) - 1.0)
    warped = np.interp(pos, np.arange(len(x), dtype=np.float64), x).astype(np.float32)
    return warped, {
        "source_preutterance_ms": float(s_pre),
        "source_consonant_ms": float(s_con),
        "target_preutterance_ms": float(t_pre),
        "target_consonant_ms": float(t_con),
        "oto_landmark_found": bool(oto_entry is not None),
    }


def _soft_highband(freqs, sr):
    f = torch.as_tensor(freqs)
    mask = torch.zeros_like(f)
    rise = (f >= LOW_HZ) & (f < FULL_HZ)
    if bool(rise.any()):
        u = (f[rise] - LOW_HZ) / max(1.0, FULL_HZ - LOW_HZ)
        mask[rise] = 0.5 - 0.5 * torch.cos(torch.pi * torch.clamp(u, 0.0, 1.0))
    hi = min(TOP_HZ, float(sr) * 0.5 - 80.0)
    mask[(f >= FULL_HZ) & (f <= hi)] = 1.0
    fall_end = min(float(sr) * 0.5, hi + 1000.0)
    fall = (f > hi) & (f < fall_end)
    if bool(fall.any()):
        u = (f[fall] - hi) / max(1.0, fall_end - hi)
        mask[fall] = 0.5 + 0.5 * torch.cos(torch.pi * torch.clamp(u, 0.0, 1.0))
    return torch.clamp(mask, 0.0, 1.0)


def _f0_frames(f0, frames):
    if f0.dim() == 1:
        f0 = f0.view(1, 1, -1)
    elif f0.dim() == 2:
        f0 = f0.unsqueeze(1)
    return F.interpolate(f0, size=int(frames), mode="linear", align_corners=False)


def _harmonic_mask(freqs, f0, frames, width=0.18):
    track = _f0_frames(f0, frames)
    voiced = (track > 1.0).to(track.dtype)
    safe = torch.clamp(track, min=55.0)
    grid = freqs.view(1, -1, 1)
    order = torch.round(grid / safe)
    distance = torch.abs(grid - order * safe) / safe
    harm = torch.exp(-0.5 * torch.pow(distance / float(width), 2.0))
    harm = harm * (order >= 1.0).to(harm.dtype) * voiced
    return torch.clamp(harm, 0.0, 1.0), voiced


def _remap_harmonic_complex(source_harm_spec, source_f0, target_f0):
    b, bins, frames = source_harm_spec.shape
    sf0 = _f0_frames(source_f0, frames).clamp_min(1.0)
    tf0 = _f0_frames(target_f0, frames).clamp_min(1.0)
    ratio = torch.where((sf0 > 1.0) & (tf0 > 1.0), sf0 / tf0, torch.ones_like(sf0))
    target_bins = torch.arange(bins, device=source_harm_spec.device, dtype=sf0.dtype).view(1, bins, 1)
    source_pos = torch.clamp(torch.round(target_bins * ratio), 0, bins - 1).long()
    if b > 1:
        source_pos = source_pos.expand(b, -1, -1)
    return torch.gather(source_harm_spec, 1, source_pos)


def _frame_activity(source_aper_spec, highband, source_voiced):
    mag = torch.log1p(36.0 * source_aper_spec.abs())
    hi = highband.view(1, -1, 1)
    den = hi.sum().clamp_min(1.0)
    energy = (mag * hi).sum(dim=1, keepdim=True) / den
    if energy.shape[-1] > 1:
        flux = torch.zeros_like(energy)
        flux[..., 1:] = torch.relu(energy[..., 1:] - energy[..., :-1])
        q = torch.quantile(flux.detach().reshape(-1), 0.75) if flux.numel() else flux.new_tensor(0.0)
        transient = torch.clamp(flux / (q + 1e-5), 0.0, 1.0)
    else:
        transient = torch.zeros_like(energy)
    voiced = source_voiced
    if voiced.shape[-1] != energy.shape[-1]:
        voiced = F.interpolate(voiced, size=energy.shape[-1], mode="nearest")
    replacement = 0.58 + 0.26 * (1.0 - voiced) + 0.16 * transient
    return torch.clamp(replacement, 0.52, 0.94), transient


def _band_rms(audio, sr, lo, hi):
    y = np.asarray(audio, dtype=np.float64).reshape(-1)
    if y.size < 32:
        return 0.0
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1.0 / float(sr))
    m = (freqs >= float(lo)) & (freqs < min(float(hi), float(sr) * 0.5))
    if not np.any(m):
        return 0.0
    band = np.fft.irfft(spec * m.astype(np.float64), n=len(y)).real
    return _rms(band)


def apply_high_detail_complex(engine, req, output_path):
    started = time.perf_counter()
    try:
        final, output_sr = sf.read(output_path, always_2d=False)
        final = _mono(final)
        if final.size < 128:
            return {"used": False, "reason": "short-output"}

        raw, source_sr = sf.read(req["input"], always_2d=False)
        raw = _mono(raw)
        raw = core.crop_oto(
            raw,
            int(source_sr),
            float(req.get("offset", 0.0)),
            float(req.get("cutoff", 0.0)),
        )
        if raw.size < 128:
            return {"used": False, "reason": "short-source"}
        if int(source_sr) != int(output_sr):
            raw = librosa.resample(raw, orig_sr=int(source_sr), target_sr=int(output_sr)).astype(np.float32)
            source_sr = int(output_sr)

        frms = max(_rms(final), 1e-8)
        srms = max(_rms(raw), 1e-8)
        raw = raw * float(np.clip(frms / srms, 0.20, 5.0))
        oto_entry = _local_oto_entry(req)
        source, landmark_stats = _landmark_warp(
            raw,
            int(source_sr),
            req,
            len(final),
            int(output_sr),
            oto_entry=oto_entry,
        )

        source_f0, target_f0 = _exact_f0_tracks(engine, req)
        base_t = torch.from_numpy(final).float().view(1, 1, -1).to(engine.device)
        source_t = torch.from_numpy(source).float().view(1, 1, -1).to(engine.device)
        window = torch.hann_window(N_FFT, device=engine.device, dtype=base_t.dtype)

        with torch.inference_mode():
            final_spec = torch.stft(
                base_t.squeeze(1), n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                window=window, return_complex=True,
            )
            source_spec = torch.stft(
                source_t.squeeze(1), n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                window=window, return_complex=True,
            )
            frames = min(final_spec.shape[-1], source_spec.shape[-1])
            final_spec = final_spec[..., :frames]
            source_spec = source_spec[..., :frames]
            freqs = torch.linspace(
                0.0, float(output_sr) * 0.5, final_spec.shape[1],
                device=engine.device, dtype=base_t.dtype,
            )
            highband = _soft_highband(freqs, int(output_sr))
            source_harm, source_voiced = _harmonic_mask(freqs, source_f0, frames, width=0.16)
            target_harm, target_voiced = _harmonic_mask(freqs, target_f0, frames, width=0.16)

            source_harm_spec = source_spec * source_harm
            source_aper_spec = source_spec * (1.0 - source_harm)
            remapped_harm = _remap_harmonic_complex(source_harm_spec, source_f0, target_f0) * target_harm

            hb = highband.view(1, -1, 1)
            final_harm = final_spec * target_harm
            final_nonharm = final_spec * (1.0 - target_harm)
            replacement, transient = _frame_activity(source_aper_spec, highband, source_voiced)
            replacement = replacement.expand(-1, final_spec.shape[1], -1) * hb

            target_voiced_f = target_voiced
            if target_voiced_f.shape[-1] != frames:
                target_voiced_f = F.interpolate(target_voiced_f, size=frames, mode="nearest")
            harmonic_mix_frame = 0.34 + 0.16 * transient
            harmonic_mix_frame = harmonic_mix_frame * target_voiced_f
            harmonic_mix = harmonic_mix_frame.expand(-1, final_spec.shape[1], -1) * hb

            transferred_nonharm = (1.0 - replacement) * final_nonharm + replacement * source_aper_spec
            transferred_harm = (1.0 - harmonic_mix) * final_harm + harmonic_mix * remapped_harm
            high_out = transferred_nonharm + transferred_harm
            out_spec = final_spec * (1.0 - hb) + high_out * hb

            out_t = torch.istft(
                out_spec,
                n_fft=N_FFT,
                hop_length=HOP,
                win_length=N_FFT,
                window=window,
                length=len(final),
            ).unsqueeze(1)

        out = out_t[0, 0].detach().cpu().numpy().astype(np.float64)
        residual = out - final.astype(np.float64)
        residual_rms = _rms(residual)
        cap = frms * 0.42
        cap_gain = 1.0
        if residual_rms > cap > 1e-9:
            cap_gain = cap / residual_rms
            out = final.astype(np.float64) + cap_gain * residual
            residual *= cap_gain
            residual_rms *= cap_gain

        peak = float(np.max(np.abs(out))) if out.size else 0.0
        peak_gain = 1.0
        if peak > 0.985:
            peak_gain = 0.975 / peak
            out *= peak_gain
        out = np.nan_to_num(out).astype(np.float32)

        path = Path(output_path)
        tmp = path.parent / f".{path.name}.complex-transfer-{os.getpid()}-{time.time_ns()}.wav"
        try:
            sf.write(tmp, out, int(output_sr), subtype="PCM_16")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

        bands = {}
        for lo, hi, label in ((4000, 8000, "4-8k"), (8000, 12000, "8-12k"), (12000, 20000, "12-20k")):
            before = _band_rms(final, int(output_sr), lo, hi)
            after = _band_rms(out, int(output_sr), lo, hi)
            bands[label] = {
                "before_rms": float(before),
                "after_rms": float(after),
                "ratio": float(after / max(before, 1e-9)),
            }

        elapsed = (time.perf_counter() - started) * 1000.0
        return {
            "used": True,
            "backend": "raw-complex-pitch-remap-replacement-v1",
            "replacement_mean": float(replacement[:, highband > 0.2, :].mean().detach().cpu()),
            "harmonic_mix_mean": float(harmonic_mix[:, highband > 0.2, :].mean().detach().cpu()),
            "transient_mean": float(transient.mean().detach().cpu()),
            "source_voiced_fraction": float(source_voiced.mean().detach().cpu()),
            "target_voiced_fraction": float(target_voiced.mean().detach().cpu()),
            "residual_percent": float(100.0 * residual_rms / max(frms, 1e-8)),
            "residual_cap_gain": float(cap_gain),
            "peak_safety_gain": float(peak_gain),
            "bands": bands,
            "runtime_ms": float(elapsed),
            **landmark_stats,
        }
    except Exception as exc:
        return {"used": False, "reason": str(exc)}
