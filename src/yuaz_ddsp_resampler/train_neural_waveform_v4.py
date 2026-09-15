#!/usr/bin/env python3
"""Conditioned-v4 trainer."""

import os
from pathlib import Path

import torch
import torch.nn.functional as F

from . import train_neural_waveform_v3 as v3
from .neural_waveform import YuazNeuralWaveformDecoder, build_neural_conditioning
from .neural_waveform_v4 import (
    SOURCE_DETAIL_CHANNELS,
    SOURCE_DETAIL_BANDS,
    SOURCE_DETAIL_N_FFT,
    SOURCE_DETAIL_HOP,
    build_pitch_invariant_source_detail,
    append_source_detail,
)
from .train_neural_waveform import read_fullband_target, stft_mag


_V3_BASE_LOSS = v3.neural_waveform_loss_v3

PRESENCE_LOW_HZ = 2000.0
PRESENCE_HIGH_HZ = 6000.0
ARTICULATION_LOW_HZ = 1000.0
ARTICULATION_HIGH_HZ = 8000.0
V4_WARM_START_ENV = "YUAZ_V4_WARM_START"


def prepare_condition_v4(engine, source, source_item, target_f0, seed, return_raw=False):
    frames = int(target_f0.shape[-1])
    latent = F.interpolate(source["latent"], size=frames, mode="linear", align_corners=False)
    detail = F.interpolate(source["detail"], size=frames, mode="linear", align_corners=False)
    context = v3.resolve_conditioning_context(engine, source_item)
    torch.manual_seed(int(seed))
    with torch.no_grad():
        raw_structure, aux = engine.decoder(
            target_f0,
            latent,
            adapter=context["adapter"],
            detail=detail,
            prototype_index=context["prototype_index"],
            timbre_shift_semitones=0.0,
            detail_strength=1.0,
            frame_controls=None,
            ai_control_adapter=context["ai_controls"],
            synthesis_sample_rate=v3.SAMPLE_RATE,
            return_aux=True,
        )

    base = build_neural_conditioning(latent, detail, target_f0, aux)
    source_samples = max(1, int(source["f0"].shape[-1]) * 256)
    source_wave = read_fullband_target(
        source_item["voicebank_root"], source_item, source_samples
    ).to(engine.device)
    source_detail = build_pitch_invariant_source_detail(source_wave, frames)
    conditioning = append_source_detail(base, source_detail)
    structure = v3.smooth_lowpass_structure(raw_structure.detach())
    if return_raw:
        return conditioning, structure, raw_structure.detach()
    return conditioning, structure


def _band_mask(n_fft, device, dtype, lo, hi):
    bins = n_fft // 2 + 1
    freqs = torch.linspace(0.0, v3.SAMPLE_RATE * 0.5, bins, device=device, dtype=dtype)
    return (freqs >= float(lo)) & (freqs < float(hi))


def _presence_logmag_loss(target, pred):
    terms = []
    for n_fft in (512, 1024, 2048):
        if target.shape[-1] < n_fft:
            continue
        t = torch.log1p(stft_mag(target, n_fft))
        p = torch.log1p(stft_mag(pred, n_fft))
        mask = _band_mask(n_fft, t.device, t.dtype, PRESENCE_LOW_HZ, PRESENCE_HIGH_HZ)
        if bool(mask.any()):
            terms.append(F.l1_loss(p[:, mask], t[:, mask]))
    return torch.stack(terms).mean() if terms else target.new_tensor(0.0)


def _articulation_flux_loss(target, pred):
    terms = []
    for n_fft in (256, 512, 1024):
        if target.shape[-1] < n_fft:
            continue
        t = torch.log1p(stft_mag(target, n_fft))
        p = torch.log1p(stft_mag(pred, n_fft))
        mask = _band_mask(n_fft, t.device, t.dtype, ARTICULATION_LOW_HZ, ARTICULATION_HIGH_HZ)
        if not bool(mask.any()) or t.shape[-1] < 2:
            continue
        dt = torch.diff(t[:, mask], dim=-1)
        dp = torch.diff(p[:, mask], dim=-1)
        terms.append(F.l1_loss(dp, dt))
    return torch.stack(terms).mean() if terms else target.new_tensor(0.0)


def _envelope_derivative_loss(target, pred):
    terms = []
    for kernel, stride in ((97, 24), (193, 48), (385, 96)):
        if target.shape[-1] < kernel:
            continue
        t = F.avg_pool1d(target.abs(), kernel, stride=stride, padding=kernel // 2)
        p = F.avg_pool1d(pred.abs(), kernel, stride=stride, padding=kernel // 2)
        if t.shape[-1] > 1:
            terms.append(F.l1_loss(torch.diff(p, dim=-1), torch.diff(t, dim=-1)))
    return torch.stack(terms).mean() if terms else target.new_tensor(0.0)


def neural_waveform_loss_v4(target, pred):
    base, base_parts = _V3_BASE_LOSS(target, pred)
    presence = _presence_logmag_loss(target, pred)
    articulation_flux = _articulation_flux_loss(target, pred)
    envelope_derivative = _envelope_derivative_loss(target, pred)

    loss = (
        base
        + 0.20 * presence
        + 0.22 * articulation_flux
        + 0.12 * envelope_derivative
    )
    parts = dict(base_parts)
    parts.update({
        "presence_2_6k": float(presence.detach()),
        "articulation_flux_1_8k": float(articulation_flux.detach()),
        "envelope_derivative": float(envelope_derivative.detach()),
    })
    return loss, parts


def make_model_v4(engine, item):
    sample = v3.load_cache(item["_cache"], engine.device)
    conditioning, _ = prepare_condition_v4(
        engine, sample, item, sample["f0"], v3.stable_seed(str(item["_cache"]), "v4-shape")
    )
    model = YuazNeuralWaveformDecoder(condition_channels=int(conditioning.shape[1])).to(engine.device)
    expected = int(round(v3.SAMPLE_RATE * engine.hop / engine.sr))
    if model.output_hop != expected:
        raise RuntimeError(
            f"neural decoder output hop {model.output_hop} does not match Yuaz frame hop {expected}"
        )

    warm = os.environ.get(V4_WARM_START_ENV, "").strip()
    if warm:
        path = Path(warm).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"v4 warm-start checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload.get("state_dict") or {}
        current = model.state_dict()
        copied = 0
        for key, value in state.items():
            if key == "pre.weight":
                dst = current[key]
                if value.shape[0] != dst.shape[0] or value.shape[2:] != dst.shape[2:]:
                    raise RuntimeError("v3/v4 pre.weight output shape mismatch")
                if value.shape[1] > dst.shape[1]:
                    raise RuntimeError("v3 conditioning is wider than v4 conditioning")
                dst[:, : value.shape[1]] = value.to(dst.dtype)
                dst[:, value.shape[1] :] *= 0.05
                current[key] = dst
                copied += 1
            elif key in current and current[key].shape == value.shape:
                current[key] = value.to(current[key].dtype)
                copied += 1
        model.load_state_dict(current, strict=True)
        print(
            f"v4 warm-started from {path} copied={copied}/{len(current)} "
            f"base_channels={state.get('pre.weight').shape[1] if state.get('pre.weight') is not None else 'unknown'} "
            f"v4_channels={conditioning.shape[1]} source_detail={SOURCE_DETAIL_CHANNELS}",
            flush=True,
        )
    return model


def metadata_v4(
    voicebank,
    manifest,
    val_aliases,
    pair_buckets,
    val_pair_buckets,
    history,
    role,
    best_native=None,
    best_pair=None,
    pareto_native_limit=None,
):
    return {
        "version": "0.3.0",
        "trainer_generation": "conditioned-v4",
        "sample_rate": v3.SAMPLE_RATE,
        "voicebank": str(voicebank),
        "manifest": str(manifest),
        "validation_aliases": sorted(val_aliases),
        "train_pair_buckets": pair_buckets,
        "validation_pair_buckets": val_pair_buckets,
        "history": list(history),
        "checkpoint_role": str(role).replace("v3", "v4"),
        "best_native_validation_loss": best_native,
        "best_multipitch_validation_loss": best_pair,
        "pareto_native_limit": pareto_native_limit,
        "structure_lowpass_hz": v3.STRUCTURE_LOWPASS_HZ,
        "structure_transition_hz": v3.STRUCTURE_TRANSITION_HZ,
        "pair_lr": v3.PAIR_LR,
        "pair_weight": v3.PAIR_WEIGHT,
        "native_rehearsal_weight": v3.NATIVE_REHEARSAL_WEIGHT,
        "source_detail_channels": SOURCE_DETAIL_CHANNELS,
        "source_detail_bands": SOURCE_DETAIL_BANDS,
        "source_detail_n_fft": SOURCE_DETAIL_N_FFT,
        "source_detail_hop": SOURCE_DETAIL_HOP,
        "source_detail_route": "frequency-smoothed source envelope + envelope delta + energy + spectral flux + flatness; no raw waveform and no source-F0 channel",
        "conditioning_route": "v3 active ai.14 conditioning + pitch-decoupled source-detail conditioning + low-pass DDSP structure",
        "training_definition": "v3 native/pair rehearsal and Pareto selection, warm-started from v3 with articulation-sensitive v4 losses",
        "validation_definition": "alias-isolated native reconstruction + held-out cross-pitch pairs by bucket",
        "loss": "v3 objective + 2-6k presence logmag + 1-8k articulation flux + multiscale envelope derivative",
        "design_reference": "WORLDLINE-R positive-reference analysis used only to choose clarity metrics/loss emphasis; not used as a training target",
        "warm_start": os.environ.get(V4_WARM_START_ENV, ""),
    }


def install_v4_overrides():
    v3.prepare_condition_v3 = prepare_condition_v4
    v3.neural_waveform_loss_v3 = neural_waveform_loss_v4
    v3.make_model_v3 = make_model_v4
    v3.metadata = metadata_v4


if __name__ == "__main__":
    install_v4_overrides()
    v3.main()
