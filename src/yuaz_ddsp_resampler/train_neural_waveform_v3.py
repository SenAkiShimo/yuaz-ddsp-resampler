#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .core import YuazDDSPResamplerEngine, stable_seed
from .neural_waveform import YuazNeuralWaveformDecoder, build_neural_conditioning, save_neural_waveform_decoder
from .train_neural_waveform import (
    SAMPLE_RATE,
    PITCH_BUCKETS,
    alias_split,
    bucket_name,
    build_manifest_index,
    build_pairs,
    load_cache,
    read_fullband_target,
    resolve_conditioning_context,
    resolve_manifest,
    stft_mag,
)


STRUCTURE_LOWPASS_HZ = 9000.0
STRUCTURE_TRANSITION_HZ = 1500.0
BAND_EDGES_HZ = (0.0, 1000.0, 4000.0, 8000.0, 12000.0, 18000.0)
PAIR_LR = 5e-5
PAIR_WEIGHT = 0.72
NATIVE_REHEARSAL_WEIGHT = 0.28
PARETO_NATIVE_DEGRADATION = 0.15


def smooth_lowpass_structure(x, cutoff_hz=STRUCTURE_LOWPASS_HZ, transition_hz=STRUCTURE_TRANSITION_HZ):
    if x.ndim == 2:
        x = x.unsqueeze(1)
    n = int(x.shape[-1])
    spec = torch.fft.rfft(x, dim=-1)
    freqs = torch.linspace(0.0, SAMPLE_RATE * 0.5, spec.shape[-1], device=x.device, dtype=x.dtype)
    cutoff = float(cutoff_hz)
    end = min(SAMPLE_RATE * 0.5, cutoff + float(transition_hz))
    mask = torch.ones_like(freqs)
    mask = torch.where(freqs >= end, torch.zeros_like(mask), mask)
    if end > cutoff:
        t = torch.clamp((freqs - cutoff) / (end - cutoff), 0.0, 1.0)
        taper = 0.5 * (1.0 + torch.cos(math.pi * t))
        mask = torch.where((freqs > cutoff) & (freqs < end), taper, mask)
    return torch.fft.irfft(spec * mask.view(1, 1, -1), n=n, dim=-1)


def prepare_condition_v3(engine, source, source_item, target_f0, seed, return_raw=False):
    frames = int(target_f0.shape[-1])
    latent = F.interpolate(source["latent"], size=frames, mode="linear", align_corners=False)
    detail = F.interpolate(source["detail"], size=frames, mode="linear", align_corners=False)
    context = resolve_conditioning_context(engine, source_item)
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
            synthesis_sample_rate=SAMPLE_RATE,
            return_aux=True,
        )
    conditioning = build_neural_conditioning(latent, detail, target_f0, aux)
    structure = smooth_lowpass_structure(raw_structure.detach())
    if return_raw:
        return conditioning, structure, raw_structure.detach()
    return conditioning, structure


def band_distribution_loss(target, pred, n_fft=2048):
    if target.shape[-1] < n_fft:
        return target.new_tensor(0.0)
    t = stft_mag(target, n_fft).pow(2).mean(dim=-1)
    p = stft_mag(pred, n_fft).pow(2).mean(dim=-1)
    freqs = torch.linspace(0.0, SAMPLE_RATE * 0.5, t.shape[-1], device=t.device, dtype=t.dtype)
    t_bands = []
    p_bands = []
    for lo, hi in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:]):
        mask = (freqs >= float(lo)) & (freqs < float(hi))
        if not bool(mask.any()):
            continue
        t_bands.append(t[..., mask].mean(dim=-1))
        p_bands.append(p[..., mask].mean(dim=-1))
    if not t_bands:
        return target.new_tensor(0.0)
    t_vec = torch.stack(t_bands, dim=-1)
    p_vec = torch.stack(p_bands, dim=-1)
    t_vec = t_vec / (t_vec.sum(dim=-1, keepdim=True) + 1e-8)
    p_vec = p_vec / (p_vec.sum(dim=-1, keepdim=True) + 1e-8)
    return F.l1_loss(torch.log(t_vec + 1e-6), torch.log(p_vec + 1e-6))


def log_rms_loss(target, pred):
    t = torch.sqrt(torch.mean(target * target, dim=-1) + 1e-8)
    p = torch.sqrt(torch.mean(pred * pred, dim=-1) + 1e-8)
    return F.l1_loss(torch.log(p + 1e-6), torch.log(t + 1e-6))


def neural_waveform_loss_v3(target, pred):
    spectral_terms = []
    convergence_terms = []
    flux_terms = []
    for n_fft in (256, 512, 1024, 2048):
        if target.shape[-1] < n_fft:
            continue
        t = stft_mag(target, n_fft)
        p = stft_mag(pred, n_fft)
        t_log = torch.log1p(t)
        p_log = torch.log1p(p)
        spectral_terms.append(F.l1_loss(p_log, t_log))
        convergence_terms.append(torch.linalg.vector_norm(p - t) / (torch.linalg.vector_norm(t) + 1e-6))
        if t.shape[-1] > 1:
            flux_terms.append(F.l1_loss(torch.diff(p_log, dim=-1), torch.diff(t_log, dim=-1)))
    if not spectral_terms:
        spectral = F.l1_loss(pred, target)
        convergence = target.new_tensor(0.0)
        flux = target.new_tensor(0.0)
    else:
        spectral = torch.stack(spectral_terms).mean()
        convergence = torch.stack(convergence_terms).mean()
        flux = torch.stack(flux_terms).mean() if flux_terms else target.new_tensor(0.0)

    envelope_kernel = 385
    t_env = F.avg_pool1d(target.abs(), envelope_kernel, stride=96, padding=envelope_kernel // 2)
    p_env = F.avg_pool1d(pred.abs(), envelope_kernel, stride=96, padding=envelope_kernel // 2)
    envelope = F.l1_loss(p_env, t_env)
    weak_wave = F.smooth_l1_loss(pred, target)
    band = band_distribution_loss(target, pred)
    loudness = log_rms_loss(target, pred)
    loss = (
        spectral
        + 0.18 * convergence
        + 0.16 * flux
        + 0.08 * envelope
        + 0.012 * weak_wave
        + 0.025 * band
        + 0.030 * loudness
    )
    return loss, {
        "spectral": float(spectral.detach()),
        "spectral_convergence": float(convergence.detach()),
        "flux": float(flux.detach()),
        "envelope": float(envelope.detach()),
        "weak_wave": float(weak_wave.detach()),
        "band_distribution": float(band.detach()),
        "log_rms": float(loudness.detach()),
    }


def make_model_v3(engine, item):
    sample = load_cache(item["_cache"], engine.device)
    conditioning, _ = prepare_condition_v3(
        engine, sample, item, sample["f0"], stable_seed(str(item["_cache"]), "v3-shape")
    )
    model = YuazNeuralWaveformDecoder(condition_channels=int(conditioning.shape[1]))
    expected = int(round(SAMPLE_RATE * engine.hop / engine.sr))
    if model.output_hop != expected:
        raise RuntimeError(f"neural decoder output hop {model.output_hop} does not match Yuaz frame hop {expected}")
    return model.to(engine.device)


def train_native_epoch_v3(engine, model, optimizer, items, limit, epoch):
    model.train()
    rng = random.Random(20260905 + epoch)
    items = list(items)
    rng.shuffle(items)
    items = items[: min(int(limit), len(items))]
    total = 0.0
    count = 0
    for idx, item in enumerate(items, 1):
        source = load_cache(item["_cache"], engine.device)
        conditioning, structure = prepare_condition_v3(
            engine, source, item, source["f0"], stable_seed(str(item["_cache"]), "v3-native", epoch)
        )
        pred = model(conditioning, structure)
        target = read_fullband_target(item["voicebank_root"], item, pred.shape[-1]).to(engine.device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = neural_waveform_loss_v3(target, pred)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        total += float(loss.detach())
        count += 1
        if idx == 1 or idx % 16 == 0 or idx == len(items):
            print(
                f"native-v3 [{idx}/{len(items)}] loss={float(loss.detach()):.4f} "
                f"spectral={parts['spectral']:.4f} band={parts['band_distribution']:.4f} "
                f"rms={parts['log_rms']:.4f}",
                flush=True,
            )
    return total / max(1, count)


def train_pair_epoch_v3(engine, model, optimizer, pairs, native_items, limit, epoch):
    model.train()
    rng = random.Random(20260905 + 100 + epoch)
    pairs = list(pairs)
    native_items = list(native_items)
    rng.shuffle(pairs)
    rng.shuffle(native_items)
    pairs = pairs[: min(int(limit), len(pairs))]
    total = 0.0
    pair_total = 0.0
    native_total = 0.0
    count = 0
    bucket_counts = {}
    for idx, pair in enumerate(pairs, 1):
        src = load_cache(pair["source"]["_cache"], engine.device)
        tgt = load_cache(pair["target"]["_cache"], engine.device)
        conditioning, structure = prepare_condition_v3(
            engine, src, pair["source"], tgt["f0"], stable_seed(pair["alias"], pair["semitones"], "v3-pair", epoch)
        )
        pred = model(conditioning, structure)
        pair_target = read_fullband_target(
            pair["target"]["voicebank_root"], pair["target"], pred.shape[-1]
        ).to(engine.device)
        pair_loss, _ = neural_waveform_loss_v3(pair_target, pred)

        anchor_item = native_items[(idx - 1) % len(native_items)]
        anchor = load_cache(anchor_item["_cache"], engine.device)
        anchor_condition, anchor_structure = prepare_condition_v3(
            engine,
            anchor,
            anchor_item,
            anchor["f0"],
            stable_seed(str(anchor_item["_cache"]), "v3-rehearsal", epoch, idx),
        )
        anchor_pred = model(anchor_condition, anchor_structure)
        anchor_target = read_fullband_target(
            anchor_item["voicebank_root"], anchor_item, anchor_pred.shape[-1]
        ).to(engine.device)
        native_loss, _ = neural_waveform_loss_v3(anchor_target, anchor_pred)

        loss = PAIR_WEIGHT * pair_loss + NATIVE_REHEARSAL_WEIGHT * native_loss
        if not torch.isfinite(loss):
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        total += float(loss.detach())
        pair_total += float(pair_loss.detach())
        native_total += float(native_loss.detach())
        count += 1
        name = bucket_name(pair["semitones"])
        bucket_counts[name] = bucket_counts.get(name, 0) + 1
        if idx == 1 or idx % 16 == 0 or idx == len(pairs):
            print(
                f"pair-v3 [{idx}/{len(pairs)}] {name} {pair['semitones']:.1f}st "
                f"mix={float(loss.detach()):.4f} pair={float(pair_loss.detach()):.4f} "
                f"native={float(native_loss.detach()):.4f}",
                flush=True,
            )
    denom = max(1, count)
    return total / denom, pair_total / denom, native_total / denom, bucket_counts


def validate_native_v3(engine, model, items, limit=48):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for item in items[: min(int(limit), len(items))]:
            sample = load_cache(item["_cache"], engine.device)
            conditioning, structure = prepare_condition_v3(
                engine, sample, item, sample["f0"], stable_seed(str(item["_cache"]), "v3-validation-native")
            )
            pred = model(conditioning, structure)
            target = read_fullband_target(item["voicebank_root"], item, pred.shape[-1]).to(engine.device)
            loss, _ = neural_waveform_loss_v3(target, pred)
            if torch.isfinite(loss):
                total += float(loss)
                count += 1
    return total / count if count else None


def validate_pairs_v3(engine, model, pairs, limit=96):
    model.eval()
    totals = {name: 0.0 for name, _, _ in PITCH_BUCKETS}
    counts = {name: 0 for name, _, _ in PITCH_BUCKETS}
    total = 0.0
    count = 0
    selected = list(pairs)[: min(int(limit), len(pairs))]
    with torch.no_grad():
        for pair in selected:
            src = load_cache(pair["source"]["_cache"], engine.device)
            tgt = load_cache(pair["target"]["_cache"], engine.device)
            conditioning, structure = prepare_condition_v3(
                engine,
                src,
                pair["source"],
                tgt["f0"],
                stable_seed(pair["alias"], pair["semitones"], "v3-validation-pair"),
            )
            pred = model(conditioning, structure)
            target = read_fullband_target(
                pair["target"]["voicebank_root"], pair["target"], pred.shape[-1]
            ).to(engine.device)
            loss, _ = neural_waveform_loss_v3(target, pred)
            if not torch.isfinite(loss):
                continue
            value = float(loss)
            name = bucket_name(pair["semitones"])
            total += value
            count += 1
            totals[name] += value
            counts[name] += 1
    buckets = {
        name: {"loss": (totals[name] / counts[name]) if counts[name] else None, "count": counts[name]}
        for name, _, _ in PITCH_BUCKETS
    }
    return {"overall": (total / count) if count else None, "count": count, "buckets": buckets}


def checkpoint_paths(output):
    output = Path(output)
    suffix = output.suffix or ".pt"
    stem = output.stem if output.suffix else output.name
    return {
        "final": output,
        "native_best": output.with_name(stem + "-native-best" + suffix),
        "multipitch_best": output.with_name(stem + "-multipitch-best" + suffix),
        "pareto_best": output.with_name(stem + "-pareto-best" + suffix),
    }


def metadata(
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
        "trainer_generation": "conditioned-v3",
        "sample_rate": SAMPLE_RATE,
        "voicebank": str(voicebank),
        "manifest": str(manifest),
        "validation_aliases": sorted(val_aliases),
        "train_pair_buckets": pair_buckets,
        "validation_pair_buckets": val_pair_buckets,
        "history": list(history),
        "checkpoint_role": str(role),
        "best_native_validation_loss": best_native,
        "best_multipitch_validation_loss": best_pair,
        "pareto_native_limit": pareto_native_limit,
        "structure_lowpass_hz": STRUCTURE_LOWPASS_HZ,
        "structure_transition_hz": STRUCTURE_TRANSITION_HZ,
        "pair_lr": PAIR_LR,
        "pair_weight": PAIR_WEIGHT,
        "native_rehearsal_weight": NATIVE_REHEARSAL_WEIGHT,
        "conditioning_route": "active ai.14 adapter + source OTO prototype + learned-control packs + low-pass DDSP structure",
        "training_definition": "native reconstruction + mixed native/pair rehearsal with held-out alias cross-pitch validation",
        "validation_definition": "alias-isolated native reconstruction + held-out cross-pitch pairs by bucket",
        "loss": "MR-STFT + spectral convergence + flux + envelope + weak waveform + band distribution + log-RMS",
    }


def save_checkpoint(path, model, **kwargs):
    save_neural_waveform_decoder(path, model, metadata(**kwargs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--voicebank", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--native-epochs", type=int, default=3)
    parser.add_argument("--pair-epochs", type=int, default=2)
    parser.add_argument("--native-limit", type=int, default=512)
    parser.add_argument("--pair-limit", type=int, default=512)
    parser.add_argument("--native-val-limit", type=int, default=48)
    parser.add_argument("--pair-val-limit", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--pair-lr", type=float, default=PAIR_LR)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    voicebank = Path(args.voicebank).expanduser().resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = resolve_manifest(voicebank, args.manifest)
    print(f"Using ai.14 manifest: {manifest}")
    print(
        f"v3 structure low-pass={STRUCTURE_LOWPASS_HZ:.0f}Hz transition={STRUCTURE_TRANSITION_HZ:.0f}Hz "
        f"pair/native={PAIR_WEIGHT:.2f}/{NATIVE_REHEARSAL_WEIGHT:.2f}"
    )

    engine = YuazDDSPResamplerEngine(
        config["yuaz_repo"], config["checkpoint"], output_sr=SAMPLE_RATE,
        registry_path=config.get("registry_path"), ddsp_synthesis_sr=SAMPLE_RATE,
    )
    entries = build_manifest_index(manifest, voicebank)
    if not entries:
        raise RuntimeError("no usable cached voicebank entries")
    train_items, val_items, val_aliases = alias_split(entries)
    train_pairs, pair_buckets = build_pairs(train_items)
    val_pairs, val_pair_buckets = build_pairs(val_items)
    if not train_items or not val_items:
        raise RuntimeError("train/validation split is empty")
    print(f"native train={len(train_items)} val={len(val_items)} held-out aliases={len(val_aliases)}")
    print(f"train pair buckets={pair_buckets}")
    print(f"validation pair buckets={val_pair_buckets}")

    model = make_model_v3(engine, train_items[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-5)
    history = []
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else root / "control_models" / "neural-waveform-v0.3.0-conditioned-v3.pt"
    )
    paths = checkpoint_paths(output)
    best_native = None
    best_pair = None
    best_pareto = None
    pareto_native_limit = None

    common = dict(
        voicebank=voicebank,
        manifest=manifest,
        val_aliases=val_aliases,
        pair_buckets=pair_buckets,
        val_pair_buckets=val_pair_buckets,
    )

    for epoch in range(int(args.native_epochs)):
        train_loss = train_native_epoch_v3(engine, model, optimizer, train_items, args.native_limit, epoch)
        native_val = validate_native_v3(engine, model, val_items, args.native_val_limit)
        row = {"stage": "native-v3", "epoch": epoch + 1, "train_loss": train_loss, "native_validation_loss": native_val}
        history.append(row)
        print(f"native-v3 epoch {epoch + 1}: train={train_loss:.4f} native_val={native_val}")
        if native_val is not None and (best_native is None or native_val < best_native):
            best_native = native_val
            pareto_native_limit = best_native * (1.0 + PARETO_NATIVE_DEGRADATION)
            save_checkpoint(
                paths["native_best"], model, history=history, role="native-best-v3",
                best_native=best_native, best_pair=best_pair, pareto_native_limit=pareto_native_limit, **common,
            )
            print(f"saved native best v3: {paths['native_best']} ({best_native:.6f})")

    for group in optimizer.param_groups:
        group["lr"] = float(args.pair_lr)
    print(f"pair stage lr={float(args.pair_lr):.6g} native_limit={pareto_native_limit}")

    for epoch in range(int(args.pair_epochs)):
        mix_loss, pair_train, native_rehearsal, seen = train_pair_epoch_v3(
            engine, model, optimizer, train_pairs, train_items, args.pair_limit, epoch
        )
        native_val = validate_native_v3(engine, model, val_items, args.native_val_limit)
        pair_val = validate_pairs_v3(engine, model, val_pairs, args.pair_val_limit)
        pair_overall = pair_val["overall"]
        eligible = (
            pair_overall is not None
            and native_val is not None
            and pareto_native_limit is not None
            and native_val <= pareto_native_limit
        )
        row = {
            "stage": "multipitch-v3",
            "epoch": epoch + 1,
            "train_loss": mix_loss,
            "pair_train_loss": pair_train,
            "native_rehearsal_loss": native_rehearsal,
            "native_validation_loss": native_val,
            "cross_pitch_validation": pair_val,
            "pareto_eligible": bool(eligible),
            "seen_buckets": seen,
        }
        history.append(row)
        print(
            f"multipitch-v3 epoch {epoch + 1}: mix={mix_loss:.4f} pair={pair_train:.4f} rehearsal={native_rehearsal:.4f} "
            f"native_val={native_val} cross_val={pair_overall} pareto={eligible} buckets={pair_val['buckets']}"
        )
        if pair_overall is not None and (best_pair is None or pair_overall < best_pair):
            best_pair = pair_overall
            save_checkpoint(
                paths["multipitch_best"], model, history=history, role="multipitch-best-v3",
                best_native=best_native, best_pair=best_pair, pareto_native_limit=pareto_native_limit, **common,
            )
            print(f"saved multipitch best v3: {paths['multipitch_best']} ({best_pair:.6f})")
        if eligible and (best_pareto is None or pair_overall < best_pareto):
            best_pareto = pair_overall
            save_checkpoint(
                paths["pareto_best"], model, history=history, role="pareto-best-v3",
                best_native=best_native, best_pair=best_pair, pareto_native_limit=pareto_native_limit, **common,
            )
            print(
                f"saved pareto best v3: {paths['pareto_best']} "
                f"cross={best_pareto:.6f} native={native_val:.6f}"
            )

    save_checkpoint(
        paths["final"], model, history=history, role="final-v3",
        best_native=best_native, best_pair=best_pair, pareto_native_limit=pareto_native_limit, **common,
    )
    print(f"saved final v3: {paths['final']}")
    if best_native is not None:
        print(f"native best v3: {paths['native_best']} loss={best_native:.6f}")
    if best_pair is not None:
        print(f"multipitch best v3: {paths['multipitch_best']} cross_val={best_pair:.6f}")
    if best_pareto is not None:
        print(f"pareto best v3: {paths['pareto_best']} cross_val={best_pareto:.6f}")
    else:
        print("pareto best v3: none (all pair checkpoints exceeded native degradation limit)")


if __name__ == "__main__":
    main()
