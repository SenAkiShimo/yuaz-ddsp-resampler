#!/usr/bin/env python3
"""v0.4 HiFi fine-tune from a voicebank-matched robust v4 checkpoint."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from . import train_neural_waveform_v3 as v3
from . import train_neural_waveform_v4 as v4
from . import train_neural_waveform_v04 as robust
from .neural_waveform import YuazNeuralWaveformDecoder
from .neural_waveform_hifi import (
    HIFI_BANDS,
    HIFI_CHANNELS,
    HIFI_HIGH_HZ,
    HIFI_HOP,
    HIFI_LOW_HZ,
    HIFI_N_FFT,
    append_highband_detail,
    build_highband_source_detail,
)
from .train_neural_waveform import (
    SAMPLE_RATE,
    alias_split,
    build_manifest_index,
    build_pairs,
    load_cache,
    read_fullband_target,
    resolve_manifest,
    stft_mag,
)


TRAINER_GENERATION = "conditioned-v4-hifi"


def prepare_condition_hifi(engine, source, source_item, target_f0, seed, return_raw=False):
    base_result = v4.prepare_condition_v4(
        engine, source, source_item, target_f0, seed, return_raw=return_raw
    )
    if return_raw:
        base, structure, raw = base_result
    else:
        base, structure = base_result

    frames = int(target_f0.shape[-1])
    source_samples = max(1, int(source["f0"].shape[-1]) * 256)
    source_wave = read_fullband_target(
        source_item["voicebank_root"], source_item, source_samples
    ).to(engine.device)
    fine = build_highband_source_detail(source_wave, frames, sample_rate=SAMPLE_RATE)
    conditioning = append_highband_detail(base, fine)
    if return_raw:
        return conditioning, structure, raw
    return conditioning, structure


def _band_mask(n_fft, device, dtype, lo, hi):
    bins = n_fft // 2 + 1
    freqs = torch.linspace(0.0, SAMPLE_RATE * 0.5, bins, device=device, dtype=dtype)
    return (freqs >= float(lo)) & (freqs < float(hi))


def _highband_logmag_loss(target, pred):
    terms = []
    # Separate presence and air bands so 6-12 kHz cannot dominate 12-20 kHz.
    for lo, hi, weight in ((6000.0, 12000.0, 1.0), (12000.0, 18000.0, 0.8), (18000.0, 22000.0, 0.35)):
        band_terms = []
        for n_fft in (1024, 2048, 4096):
            if target.shape[-1] < n_fft:
                continue
            t = torch.log1p(stft_mag(target, n_fft))
            p = torch.log1p(stft_mag(pred, n_fft))
            mask = _band_mask(n_fft, t.device, t.dtype, lo, hi)
            if bool(mask.any()):
                band_terms.append(F.l1_loss(p[:, mask], t[:, mask]))
        if band_terms:
            terms.append(float(weight) * torch.stack(band_terms).mean())
    return torch.stack(terms).sum() if terms else target.new_tensor(0.0)


def _highband_flux_loss(target, pred):
    terms = []
    for n_fft in (512, 1024, 2048):
        if target.shape[-1] < n_fft:
            continue
        t = torch.log1p(stft_mag(target, n_fft))
        p = torch.log1p(stft_mag(pred, n_fft))
        mask = _band_mask(n_fft, t.device, t.dtype, 5000.0, 18000.0)
        if bool(mask.any()) and t.shape[-1] > 1:
            terms.append(
                F.l1_loss(
                    torch.diff(p[:, mask], dim=-1),
                    torch.diff(t[:, mask], dim=-1),
                )
            )
    return torch.stack(terms).mean() if terms else target.new_tensor(0.0)


def neural_waveform_loss_hifi(target, pred):
    base, parts = v4.neural_waveform_loss_v4(target, pred)
    highband = _highband_logmag_loss(target, pred)
    highflux = _highband_flux_loss(target, pred)
    loss = base + 0.32 * highband + 0.18 * highflux
    out = dict(parts)
    out.update({
        "highband_logmag_6_22k": float(highband.detach()),
        "highband_flux_5_18k": float(highflux.detach()),
    })
    return loss, out


def _load_wider_model(engine, item, checkpoint):
    sample = load_cache(item["_cache"], engine.device)
    conditioning, _ = prepare_condition_hifi(
        engine, sample, item, sample["f0"], v3.stable_seed(str(item["_cache"]), "hifi-shape")
    )
    model = YuazNeuralWaveformDecoder(condition_channels=int(conditioning.shape[1])).to(engine.device)
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    state = payload.get("state_dict") or {}
    current = model.state_dict()
    copied = 0
    for key, value in state.items():
        if key == "pre.weight":
            dst = current[key]
            if value.shape[0] != dst.shape[0] or value.shape[2:] != dst.shape[2:]:
                raise RuntimeError("v4/HiFi pre.weight output shape mismatch")
            if value.shape[1] > dst.shape[1]:
                raise RuntimeError("v4 conditioning is wider than HiFi conditioning")
            dst[:, : value.shape[1]] = value.to(dst.dtype)
            # Start fine-detail channels near zero so the first HiFi step behaves
            # almost exactly like the already-good robust v4 checkpoint.
            dst[:, value.shape[1] :] *= 0.02
            current[key] = dst
            copied += 1
        elif key in current and current[key].shape == value.shape:
            current[key] = value.to(current[key].dtype)
            copied += 1
    model.load_state_dict(current, strict=True)
    print(
        f"HiFi warm-started from {checkpoint} copied={copied}/{len(current)} "
        f"base_channels={state['pre.weight'].shape[1]} hifi_channels={conditioning.shape[1]} "
        f"added_highband={HIFI_CHANNELS}",
        flush=True,
    )
    return model


def metadata_hifi(
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
        "version": "0.4.0",
        "trainer_generation": TRAINER_GENERATION,
        "sample_rate": SAMPLE_RATE,
        "voicebank": str(Path(voicebank).expanduser().resolve()),
        "manifest": str(manifest),
        "validation_aliases": sorted(val_aliases),
        "train_pair_buckets": pair_buckets,
        "validation_pair_buckets": val_pair_buckets,
        "history": list(history),
        "checkpoint_role": str(role),
        "best_native_validation_loss": best_native,
        "best_multipitch_validation_loss": best_pair,
        "pareto_native_limit": pareto_native_limit,
        "hifi_source_detail_channels": HIFI_CHANNELS,
        "hifi_source_detail_bands": HIFI_BANDS,
        "hifi_source_detail_n_fft": HIFI_N_FFT,
        "hifi_source_detail_hop": HIFI_HOP,
        "hifi_source_detail_low_hz": HIFI_LOW_HZ,
        "hifi_source_detail_high_hz": HIFI_HIGH_HZ,
        "conditioning_route": "robust conditioned-v4 + gated fine 6-22k source texture",
        "loss": "v4 objective + 6-22k logmag + 5-18k high-band flux",
        "training_definition": "short same-voicebank HiFi fine-tune from robust v4 Pareto checkpoint",
        "warm_start": os.environ.get("YUAZ_HIFI_WARM_START", ""),
    }


def _verify_warm_start(path, voicebank):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    meta = dict(payload.get("metadata") or {})
    trained = Path(str(meta.get("voicebank") or "")).expanduser().resolve()
    requested = Path(voicebank).expanduser().resolve()
    if trained != requested:
        raise RuntimeError(f"HiFi warm-start voicebank mismatch: {trained} != {requested}")
    generation = str(meta.get("trainer_generation") or "")
    if generation != "conditioned-v4":
        raise RuntimeError(f"HiFi requires conditioned-v4 warm start, got {generation or 'missing'}")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--voicebank", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--warm-start", default=None)
    parser.add_argument("--native-epochs", type=int, default=1)
    parser.add_argument("--pair-epochs", type=int, default=2)
    parser.add_argument("--native-limit", type=int, default=384)
    parser.add_argument("--pair-limit", type=int, default=512)
    parser.add_argument("--native-val-limit", type=int, default=48)
    parser.add_argument("--pair-val-limit", type=int, default=96)
    parser.add_argument("--min-train-per-bucket", type=int, default=128)
    parser.add_argument("--min-val-per-bucket", type=int, default=24)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--pair-lr", type=float, default=3e-5)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    voicebank = Path(args.voicebank).expanduser().resolve()
    manifest = resolve_manifest(voicebank, args.manifest)
    bank_id = robust.voicebank_id(voicebank)
    bank_root = root / "control_models" / "v0.4" / bank_id
    warm = Path(args.warm_start).expanduser().resolve() if args.warm_start else (
        bank_root / "v4" / "neural-waveform-v0.4-v4-pareto-best.pt"
    )
    if not warm.is_file():
        raise RuntimeError(f"robust v4 Pareto warm-start checkpoint not found: {warm}")
    _verify_warm_start(warm, voicebank)
    os.environ["YUAZ_HIFI_WARM_START"] = str(warm)

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    engine = v3.YuazDDSPResamplerEngine(
        config["yuaz_repo"], config["checkpoint"], output_sr=SAMPLE_RATE,
        registry_path=config.get("registry_path"), ddsp_synthesis_sr=SAMPLE_RATE,
    )
    entries = build_manifest_index(manifest, voicebank)
    train_items, val_items, val_aliases = alias_split(entries)
    if not train_items or not val_items:
        raise RuntimeError("train/validation split is empty")

    recorded_train, _ = build_pairs(train_items)
    recorded_val, _ = build_pairs(val_items)
    robust_train = robust.top_up_pairs(train_items, recorded_train, args.min_train_per_bucket, 20260917)
    robust_val = robust.top_up_pairs(val_items, recorded_val, args.min_val_per_bucket, 20260918)
    train_buckets = robust.pair_bucket_counts(robust_train)
    val_buckets = robust.pair_bucket_counts(robust_val)
    train_modes = robust.pair_mode_counts(robust_train)
    val_modes = robust.pair_mode_counts(robust_val)

    model = _load_wider_model(engine, train_items[0], warm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-5)
    hifi_dir = bank_root / "hifi"
    hifi_dir.mkdir(parents=True, exist_ok=True)
    paths = robust.checkpoint_paths(hifi_dir, "neural-waveform-v0.4-hifi")

    # Native helpers in the established v3 trainer use module-level routes.
    old_prepare = v3.prepare_condition_v3
    old_loss = v3.neural_waveform_loss_v3
    try:
        v3.prepare_condition_v3 = prepare_condition_hifi
        v3.neural_waveform_loss_v3 = neural_waveform_loss_hifi
        pareto, score = robust.run_stage(
            engine=engine,
            model=model,
            optimizer=optimizer,
            train_items=train_items,
            val_items=val_items,
            robust_train=robust_train,
            robust_val=robust_val,
            args=args,
            stage_name="hifi",
            prepare_fn=prepare_condition_hifi,
            loss_fn=neural_waveform_loss_hifi,
            metadata_builder=metadata_hifi,
            paths=paths,
            bank_id=bank_id,
            voicebank=voicebank,
            manifest=manifest,
            val_aliases=val_aliases,
            train_pair_buckets=train_buckets,
            val_pair_buckets=val_buckets,
            train_modes=train_modes,
            val_modes=val_modes,
        )
    finally:
        v3.prepare_condition_v3 = old_prepare
        v3.neural_waveform_loss_v3 = old_loss

    robust.validate_checkpoint_voicebank(pareto, voicebank, bank_id)
    record = bank_root / "recommended-hifi.json"
    record.write_text(json.dumps({
        "version": "0.4.0",
        "voicebank": str(voicebank),
        "voicebank_id": bank_id,
        "checkpoint": str(pareto),
        "warm_start": str(warm),
        "cross_pitch_score": score,
        "highband_route": "gated fine 6-22k source texture",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"v0.4 HiFi training complete for {bank_id}")
    print(f"recommended HiFi checkpoint: {pareto}")
    print(f"selection record: {record}")


if __name__ == "__main__":
    main()
