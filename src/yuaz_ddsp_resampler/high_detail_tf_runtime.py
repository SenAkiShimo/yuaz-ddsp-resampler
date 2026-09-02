#!/usr/bin/env python3
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from . import core
from .high_detail_tf_generator import load_high_detail_tf
from .source_high_detail import (
    CACHE_SR,
    _find_cache,
    _piecewise_warp_detail,
)


MODEL_NAME = "high_detail_tf.pt"
_MODEL_CACHE = {}


def _rms(x):
    y = np.asarray(x, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(y * y) + 1e-12)) if y.size else 0.0


def _mono(x):
    y = np.asarray(x)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return np.nan_to_num(np.asarray(y, dtype=np.float32).reshape(-1))


def _find_model(input_path):
    source = Path(input_path).expanduser().resolve()
    for parent in (source.parent, *source.parents):
        candidate = parent / ".yuaz-0.2.8ai14" / MODEL_NAME
        if candidate.is_file():
            return candidate
    return None


def _load_model(path, device):
    path = Path(path).resolve()
    key = str(path)
    mtime = path.stat().st_mtime_ns
    cached = _MODEL_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1], cached[2]
    model, meta = load_high_detail_tf(path, device=device)
    _MODEL_CACHE[key] = (mtime, model, meta)
    if len(_MODEL_CACHE) > 8:
        _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
    return model, meta


def _exact_f0_tracks(engine, req):
    a = engine.analyze(req)
    source_f0_t = a["f0"].to(engine.device)

    velocity = max(1.0, float(req["velocity"]))
    stretch_ratio = 2.0 ** (1.0 - velocity * 0.01)
    source_fixed_ms = max(0.0, float(req["consonant"]))
    fixed_region_ms = max(0.0, source_fixed_ms * stretch_ratio)
    target_ms = max(50.0, float(req["length"]))
    target_frames = max(4, int(round(target_ms * engine.sr / (1000.0 * engine.hop))))
    source_fixed_frames = int(round(source_fixed_ms * engine.sr / (1000.0 * engine.hop)))
    target_fixed_frames = int(round(fixed_region_ms * engine.sr / (1000.0 * engine.hop)))

    f0_warp = core.piecewise_warp(
        source_f0_t,
        source_fixed_frames,
        target_frames,
        target_fixed_frames=target_fixed_frames,
    )
    src_f0_raw = f0_warp[0, 0].detach().cpu().numpy().astype(np.float32)
    voiced_mask, first_voiced_frame, _ = core.derive_voiced_mask(src_f0_raw)
    src_f0 = core.fill_f0(src_f0_raw, first_voiced_frame)
    target_f0, _, _, _ = core.build_target_f0(
        req["tone"],
        req.get("pitch", "AA"),
        req.get("tempo", "!120"),
        target_frames,
        engine.sr,
        engine.hop,
        src_f0,
        req.get("modulation", 0),
        voiced_mask=voiced_mask,
    )
    return (
        f0_warp.detach().to(engine.device),
        torch.from_numpy(target_f0).float().view(1, 1, -1).to(engine.device),
    )


def _pitch_safety_gate(model, freqs, source_f0, target_f0, frames):
    source_harm = model._harmonic_map(freqs, source_f0, frames)
    target_harm = model._harmonic_map(freqs, target_f0, frames)
    target_absence = torch.clamp(1.0 - 1.45 * target_harm, 0.0, 1.0)
    danger = torch.clamp(source_harm * target_absence, 0.0, 1.0)
    return torch.clamp(1.0 - 0.94 * danger, 0.06, 1.0), danger


def apply_high_detail_tf(engine, req, output_path):
    model_path = _find_model(req.get("input", ""))
    if model_path is None:
        return {"used": False, "reason": "no-voicebank-tf-model"}

    _, cache_file, record = _find_cache(req.get("input", ""))
    if cache_file is None:
        return {"used": False, "reason": "no-valid-source-detail-cache"}

    started = time.perf_counter()
    try:
        final, output_sr = sf.read(output_path, always_2d=False)
        final = _mono(final)
        if final.size < 64:
            return {"used": False, "reason": "short-output"}

        model, meta = _load_model(model_path, engine.device)
        if int(model.sample_rate) != int(output_sr):
            return {
                "used": False,
                "reason": f"model-sr-mismatch:{model.sample_rate}!={output_sr}",
            }

        detail_mm = np.load(cache_file, mmap_mode="r", allow_pickle=False)
        detail = np.asarray(detail_mm, dtype=np.float32)
        cache_sr = int(record.get("cache_sr", CACHE_SR))
        detail = core.crop_oto(
            detail,
            cache_sr,
            float(req.get("offset", 0.0)),
            float(req.get("cutoff", 0.0)),
        )
        if detail.size < 32:
            return {"used": False, "reason": "short-cache-crop"}

        warped = _piecewise_warp_detail(
            detail,
            cache_sr,
            req,
            len(final),
            int(output_sr),
        )
        base_rms = max(_rms(final), 1e-8)
        detail_rms = max(_rms(warped), 1e-8)
        warped *= float(np.clip((0.12 * base_rms) / detail_rms, 0.20, 4.0))

        source_f0, target_f0 = _exact_f0_tracks(engine, req)
        base_t = torch.from_numpy(final).float().view(1, 1, -1).to(engine.device)
        source_t = torch.from_numpy(warped).float().view(1, 1, -1).to(engine.device)

        with torch.inference_mode():
            _, _, mask, src_spec, _ = model(base_t, source_t, source_f0, target_f0)
            frames = min(mask.shape[-1], src_spec.shape[-1])
            mask = mask[..., :frames]
            src_spec = src_spec[..., :frames]
            freqs = torch.linspace(
                0.0,
                float(output_sr) * 0.5,
                src_spec.shape[1],
                device=engine.device,
                dtype=base_t.dtype,
            )
            safety, danger = _pitch_safety_gate(
                model, freqs, source_f0, target_f0, frames
            )
            safe_mask = mask * safety
            residual_spec = src_spec * safe_mask
            residual = model._istft(residual_spec, len(final), base_t)

            base_rms_t = torch.sqrt(torch.mean(base_t.pow(2), dim=-1, keepdim=True) + 1e-8)
            residual_rms_t = torch.sqrt(torch.mean(residual.pow(2), dim=-1, keepdim=True) + 1e-8)
            limit = 0.30 * base_rms_t + 1e-7
            cap = torch.clamp(limit / (residual_rms_t + 1e-8), max=1.0)
            residual = residual * cap
            out_t = base_t + residual

        out = out_t[0, 0].detach().cpu().numpy().astype(np.float64)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        peak_gain = 1.0
        if peak > 0.985:
            peak_gain = 0.975 / peak
            out *= peak_gain
        out = np.nan_to_num(out).astype(np.float32)

        path = Path(output_path)
        tmp = path.parent / f".{path.name}.high-detail-tf-{os.getpid()}-{time.time_ns()}.wav"
        try:
            sf.write(tmp, out, int(output_sr), subtype="PCM_16")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

        residual_np = residual[0, 0].detach().cpu().numpy()
        high_band = freqs >= 7200.0
        high_mask = safe_mask[:, high_band, :]
        danger_high = danger[:, high_band, :]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "used": True,
            "backend": "voicebank-high-detail-tf-v1-pitch-safe",
            "model": str(model_path),
            "training_format": int(meta.get("training_format", 0) or 0),
            "best_epoch": int(meta.get("best_epoch", 0) or 0),
            "best_validation_loss": float(meta.get("best_validation_loss", 0.0) or 0.0),
            "mask_mean": float(high_mask.mean().detach().cpu()) if high_mask.numel() else 0.0,
            "mask_active": float((high_mask > 0.20).float().mean().detach().cpu()) if high_mask.numel() else 0.0,
            "pitch_danger_mean": float(danger_high.mean().detach().cpu()) if danger_high.numel() else 0.0,
            "residual_rms": float(_rms(residual_np)),
            "residual_percent": float(100.0 * _rms(residual_np) / max(base_rms, 1e-8)),
            "residual_cap_gain": float(cap.mean().detach().cpu()),
            "peak_safety_gain": float(peak_gain),
            "runtime_ms": float(elapsed_ms),
        }
    except Exception as exc:
        return {"used": False, "reason": str(exc)}
