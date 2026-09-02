#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn.functional as F

from . import train_high_detail_router_entry as entry

base = entry.base
_ORIGINAL_PREPARE = base._prepare_base


def _piecewise_landmark_warp(source, source_sr, source_entry, target_entry, target_samples, target_sr):
    x = np.asarray(source, dtype=np.float32).reshape(-1)
    if len(x) < 2 or target_samples < 2:
        return np.zeros(max(1, int(target_samples)), dtype=np.float32)

    s_pre = max(0.0, float(source_entry.preutterance))
    s_con = max(s_pre, float(source_entry.consonant))
    t_pre = max(0.0, float(target_entry.preutterance))
    t_con = max(t_pre, float(target_entry.consonant))
    s_total = len(x) * 1000.0 / float(source_sr)
    t_total = target_samples * 1000.0 / float(target_sr)

    s_pts = np.array([0.0, min(s_pre, s_total), min(s_con, s_total), s_total], dtype=np.float64)
    t_pts = np.array([0.0, min(t_pre, t_total), min(t_con, t_total), t_total], dtype=np.float64)
    for arr in (s_pts, t_pts):
        for i in range(1, len(arr)):
            if arr[i] <= arr[i - 1]:
                arr[i] = arr[i - 1] + 1e-3

    tt = np.arange(int(target_samples), dtype=np.float64) * 1000.0 / float(target_sr)
    st = np.interp(tt, t_pts, s_pts)
    pos = np.clip(st * float(source_sr) / 1000.0, 0.0, len(x) - 1.0)
    return np.interp(pos, np.arange(len(x), dtype=np.float64), x).astype(np.float32)


def _prepare_v3(engine, source_entry, target_entry):
    sample = _ORIGINAL_PREPARE(engine, source_entry, target_entry)
    if sample is None:
        return None

    target_native, target_sr = base.sf.read(target_entry.wav_path, always_2d=False)
    if getattr(target_native, "ndim", 1) > 1:
        target_native = np.mean(target_native, axis=1)
    target_native = np.nan_to_num(np.asarray(target_native, dtype=np.float32).reshape(-1))
    target_native = base.crop_oto(
        target_native, int(target_sr), target_entry.offset, target_entry.cutoff
    )
    target_detail, _ = base.extract_source_high_detail(
        target_native, int(target_sr), engine.output_sr
    )

    source_native, source_sr = base.sf.read(source_entry.wav_path, always_2d=False)
    if getattr(source_native, "ndim", 1) > 1:
        source_native = np.mean(source_native, axis=1)
    source_native = np.nan_to_num(np.asarray(source_native, dtype=np.float32).reshape(-1))
    source_native = base.crop_oto(
        source_native, int(source_sr), source_entry.offset, source_entry.cutoff
    )
    source_detail, _ = base.extract_source_high_detail(
        source_native, int(source_sr), engine.output_sr
    )

    n = int(sample["base"].shape[-1])
    source_detail = _piecewise_landmark_warp(
        source_detail, engine.output_sr, source_entry, target_entry, n, engine.output_sr
    )
    if len(target_detail) != n:
        target_detail = np.interp(
            np.linspace(0.0, 1.0, n),
            np.linspace(0.0, 1.0, max(1, len(target_detail))),
            target_detail if len(target_detail) else np.zeros(1, dtype=np.float32),
        ).astype(np.float32)

    base_rms = torch.sqrt(torch.mean(sample["base"].pow(2)) + 1e-8)
    src = torch.from_numpy(source_detail).float().view(1, 1, -1).to(engine.device)
    tgt_detail = torch.from_numpy(target_detail).float().view(1, 1, -1).to(engine.device)

    src_rms = torch.sqrt(torch.mean(src.pow(2)) + 1e-8)
    tgt_rms = torch.sqrt(torch.mean(tgt_detail.pow(2)) + 1e-8)
    src = src * torch.clamp((0.12 * base_rms) / src_rms, 0.20, 4.0)
    tgt_detail = tgt_detail * torch.clamp((0.12 * base_rms) / tgt_rms, 0.20, 4.0)

    sample["source_high"] = src
    sample["target_detail"] = tgt_detail
    sample["teacher"] = torch.clamp(sample["base"] + tgt_detail, -1.2, 1.2)
    return sample


def _detail_loss(pred, teacher, target_detail, source_f0, target_f0, sr):
    p, freqs = base._stft_mag(pred, sr)
    t, _ = base._stft_mag(teacher, sr)
    hi = min(20000.0, sr * 0.5 - 100.0)
    band = (freqs >= 7200.0) & (freqs <= hi)
    if not bool(band.any()):
        return F.smooth_l1_loss(pred, teacher), {}

    pb = p[:, band, :]
    tb = t[:, band, :]
    spectral = F.smooth_l1_loss(pb, tb)
    under = torch.relu(tb - pb)
    missing = torch.mean(under.pow(2))

    if pb.shape[-1] > 1:
        flux = F.smooth_l1_loss(torch.diff(pb, dim=-1), torch.diff(tb, dim=-1))
    else:
        flux = pred.new_tensor(0.0)

    residual = pred - (teacher - target_detail)
    td = target_detail[..., :residual.shape[-1]]
    residual_mag, _ = base._stft_mag(residual, sr)
    td_mag, _ = base._stft_mag(td, sr)
    detail_match = F.smooth_l1_loss(residual_mag[:, band, :], td_mag[:, band, :])

    loss = spectral + 1.35 * missing + 0.35 * flux + 0.90 * detail_match
    return loss, {
        "spectral": float(spectral.detach()),
        "missing": float(missing.detach()),
        "flux": float(flux.detach()),
        "detail_match": float(detail_match.detach()),
        "source_f0_leak": 0.0,
        "shape": 0.0,
        "bands": 0.0,
    }


def _evaluate_v3(model, samples, sr):
    if not samples:
        return {}
    rows = []
    model.eval()
    with torch.inference_mode():
        for s in samples:
            pred, residual, inject, suppress, _ = model(
                s["base"], s["source_high"], s["source_f0"], s["target_f0"]
            )
            base_loss, _ = _detail_loss(
                s["base"], s["teacher"], s["target_detail"], s["source_f0"], s["target_f0"], sr
            )
            pred_loss, parts = _detail_loss(
                pred, s["teacher"], s["target_detail"], s["source_f0"], s["target_f0"], sr
            )
            rows.append({
                "base": float(base_loss), "pred": float(pred_loss),
                "inject": float(inject.mean()), "suppress": float(suppress.mean()),
                "residual_percent": 100.0 * float(torch.sqrt(torch.mean(residual.pow(2)) + 1e-12)) /
                    max(float(torch.sqrt(torch.mean(s["base"].pow(2)) + 1e-12)), 1e-8),
                **parts,
            })
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    b, p = mean("base"), mean("pred")
    return {
        "base_loss": b,
        "refined_loss": p,
        "improvement_percent": 100.0 * (b - p) / max(abs(b), 1e-8),
        "inject_mean": mean("inject"),
        "suppress_mean": mean("suppress"),
        "residual_percent": mean("residual_percent"),
        "source_f0_leak": 0.0,
        "spectral": mean("spectral"),
        "flux": mean("flux"),
        "shape": 0.0,
        "bands": 0.0,
    }


base._prepare_base = _prepare_v3
base.highband_loss = lambda pred, target, source_f0, target_f0, sr: _detail_loss(
    pred,
    target if hasattr(target, "shape") else target,
    torch.zeros_like(pred),
    source_f0,
    target_f0,
    sr,
)
base.evaluate = _evaluate_v3
base.TRAINING_FORMAT = 3


if __name__ == "__main__":
    # Patch training loop targets by replacing each sample's target with the v3 teacher.
    original_prepare = base._prepare_base
    original_main = base.main

    # The legacy main calls highband_loss(pred, sample["target"], ...). Make target
    # point to teacher after preparation while preserving the real recording only
    # for diagnostics if needed later.
    def wrapped_prepare(engine, source_entry, target_entry):
        s = original_prepare(engine, source_entry, target_entry)
        if s is not None:
            s["real_target"] = s["target"]
            s["target"] = s["teacher"]
        return s

    base._prepare_base = wrapped_prepare
    original_main()
