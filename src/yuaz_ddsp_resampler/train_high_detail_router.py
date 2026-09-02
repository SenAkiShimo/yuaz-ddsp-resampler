#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from .core import (
    YuazDDSPResamplerEngine,
    crop_oto,
    deterministic_decode_dualrate,
    extract_detail_features,
    extract_f0,
    read_audio,
    resample_exact,
    stable_seed,
)
from .high_detail_router import HighDetailRouter, save_high_detail_router
from .source_high_detail import extract_source_high_detail
from .voicebank import annotate_utau_subbanks, scan_voicebank


TRAINING_FORMAT = 1


def load_config(root):
    path = Path(root) / "config.json"
    if not path.exists():
        raise RuntimeError("Run configure-macos.command first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _rms(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(x * x) + 1e-12)) if x.size else 0.0


def _exact_np(x, n):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(n)
    if x.size < n:
        return np.pad(x, (0, n - x.size)).astype(np.float32)
    return x[:n].astype(np.float32)


def _native_crop(entry, target_sr):
    y, sr = sf.read(entry.wav_path, always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = np.mean(y, axis=1)
    y = np.nan_to_num(np.asarray(y, dtype=np.float32).reshape(-1))
    y = crop_oto(y, int(sr), float(entry.offset), float(entry.cutoff))
    if int(sr) != int(target_sr):
        y = librosa.resample(y, orig_sr=int(sr), target_sr=int(target_sr)).astype(np.float32)
    return y


def _median_f0(engine, entry):
    audio = read_audio(entry.wav_path, engine.sr)
    audio = crop_oto(audio, engine.sr, entry.offset, entry.cutoff)
    if len(audio) < int(0.08 * engine.sr):
        return 0.0
    f0 = extract_f0(audio, engine.sr, engine.hop)
    voiced = f0[f0 > 1.0]
    return float(np.median(voiced)) if voiced.size else 0.0


def build_multipitch_pairs(engine, voicebank, limit):
    scan = scan_voicebank(voicebank)
    entries = list(scan["entries"])
    if not entries:
        raise RuntimeError("No usable OTO entries found.")

    manifest = []
    entry_by_key = {}
    print(f"Scanning pitch metadata for {len(entries)} OTO entries...", flush=True)
    for i, entry in enumerate(entries, 1):
        f0 = _median_f0(engine, entry)
        item = {
            "status": "ok",
            "relative_wav": entry.relative_wav,
            "alias": entry.alias,
            "median_f0_hz": f0,
            "entry_key": i - 1,
        }
        manifest.append(item)
        entry_by_key[i - 1] = entry
        if i == 1 or i % 100 == 0 or i == len(entries):
            print(f"pitch scan [{i}/{len(entries)}]", flush=True)

    subbanks = annotate_utau_subbanks(voicebank, manifest, scan["prefix_map"])
    groups = {}
    for item in manifest:
        f0 = float(item.get("median_f0_hz", 0.0) or 0.0)
        if f0 <= 0.0:
            continue
        base_alias = str(item.get("base_alias") or item.get("alias") or "").strip()
        subbank = int(item.get("subbank_index", -1))
        if not base_alias or subbank < 0:
            continue
        groups.setdefault(base_alias, []).append(item)

    candidates = []
    for alias, items in groups.items():
        by_subbank = {}
        for item in items:
            idx = int(item.get("subbank_index", -1))
            current = by_subbank.get(idx)
            if current is None:
                by_subbank[idx] = item
            else:
                old = float(current.get("median_f0_hz", 0.0))
                new = float(item.get("median_f0_hz", 0.0))
                anchor = float(item.get("subbank_anchor_midi", 60.0) or 60.0)
                anchor_hz = 440.0 * (2.0 ** ((anchor - 69.0) / 12.0))
                if abs(math.log2(max(new, 1.0) / anchor_hz)) < abs(math.log2(max(old, 1.0) / anchor_hz)):
                    by_subbank[idx] = item
        reps = sorted(by_subbank.values(), key=lambda x: float(x.get("median_f0_hz", 0.0)))
        if len(reps) < 2:
            continue
        pair_specs = []
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                a = reps[i]
                b = reps[j]
                fa = float(a["median_f0_hz"])
                fb = float(b["median_f0_hz"])
                semitones = abs(12.0 * math.log2(fb / fa))
                if semitones < 1.5:
                    continue
                pair_specs.append((semitones, a, b))
        pair_specs.sort(key=lambda x: x[0], reverse=True)
        for semitones, a, b in pair_specs[:2]:
            candidates.append({
                "alias": alias,
                "source": a,
                "target": b,
                "semitones": semitones,
            })
            candidates.append({
                "alias": alias,
                "source": b,
                "target": a,
                "semitones": semitones,
            })

    rng = random.Random(20260902)
    rng.shuffle(candidates)
    candidates.sort(key=lambda x: float(x["semitones"]), reverse=True)
    if limit > 0:
        candidates = candidates[: min(int(limit), len(candidates))]

    for pair in candidates:
        pair["source_entry"] = entry_by_key[int(pair["source"]["entry_key"])]
        pair["target_entry"] = entry_by_key[int(pair["target"]["entry_key"])]

    return candidates, {
        "oto_entries": len(entries),
        "subbanks": int(subbanks.get("prototype_count", 0)),
        "aliases_with_pitch_pairs": sum(1 for x in groups.values() if len({int(i.get('subbank_index', -1)) for i in x}) >= 2),
        "candidate_pairs": len(candidates),
    }


def _prepare_base(engine, source_entry, target_entry):
    source_audio = read_audio(source_entry.wav_path, engine.sr)
    source_audio = crop_oto(source_audio, engine.sr, source_entry.offset, source_entry.cutoff)
    target_audio = read_audio(target_entry.wav_path, engine.sr)
    target_audio = crop_oto(target_audio, engine.sr, target_entry.offset, target_entry.cutoff)
    if len(source_audio) < int(0.08 * engine.sr) or len(target_audio) < int(0.08 * engine.sr):
        return None

    source_f0_np = extract_f0(source_audio, engine.sr, engine.hop)
    target_f0_np = extract_f0(target_audio, engine.sr, engine.hop)
    if not np.any(source_f0_np > 1.0) or not np.any(target_f0_np > 1.0):
        return None

    source_t = torch.from_numpy(source_audio).float().view(1, 1, -1).to(engine.device)
    source_f0_t = torch.from_numpy(source_f0_np).float().view(1, 1, -1).to(engine.device)
    source_detail_np = extract_detail_features(source_audio, engine.sr, engine.hop)
    source_detail_t = torch.from_numpy(source_detail_np).float().unsqueeze(0).to(engine.device)

    with torch.inference_mode():
        latent, source_f0_aligned = engine.encoder(source_t, f0_override=source_f0_t)

    target_frames = max(4, len(target_f0_np))
    latent_warp = F.interpolate(latent, size=target_frames, mode="linear", align_corners=False)
    detail_warp = F.interpolate(source_detail_t, size=target_frames, mode="linear", align_corners=False)
    target_f0_t = torch.from_numpy(target_f0_np).float().view(1, 1, -1).to(engine.device)
    target_f0_t = F.interpolate(target_f0_t, size=target_frames, mode="linear", align_corners=False)
    source_f0_warp = F.interpolate(source_f0_aligned, size=target_frames, mode="linear", align_corners=False)

    adapter, _, ai_pack, record = engine._models_for_input(source_entry.wav_path)
    prototype_index = record.get("subbank_index") if record else None
    seed = stable_seed(
        str(source_entry.wav_path), str(target_entry.wav_path), "high-detail-router-v1"
    )
    _, fullband, _ = deterministic_decode_dualrate(
        engine.decoder,
        target_f0_t,
        latent_warp,
        seed,
        engine.ddsp_synthesis_sr,
        adapter=adapter,
        detail=detail_warp,
        prototype_index=prototype_index,
        ai_control_adapter=ai_pack,
    )

    target_native = _native_crop(target_entry, engine.output_sr)
    exact_samples = len(target_native)
    base = resample_exact(fullband, engine.ddsp_synthesis_sr, engine.output_sr, exact_samples)

    source_native, source_sr = sf.read(source_entry.wav_path, always_2d=False)
    if getattr(source_native, "ndim", 1) > 1:
        source_native = np.mean(source_native, axis=1)
    source_native = np.nan_to_num(np.asarray(source_native, dtype=np.float32).reshape(-1))
    source_native = crop_oto(source_native, int(source_sr), source_entry.offset, source_entry.cutoff)
    source_high, high_stats = extract_source_high_detail(source_native, int(source_sr), engine.output_sr)
    if len(source_high) < 16:
        return None
    source_high = librosa.resample(
        source_high,
        orig_sr=engine.output_sr,
        target_sr=engine.output_sr,
    ).astype(np.float32) if False else source_high.astype(np.float32)
    source_high = np.interp(
        np.linspace(0.0, 1.0, exact_samples),
        np.linspace(0.0, 1.0, len(source_high)),
        source_high,
    ).astype(np.float32)

    base_rms = max(_rms(base), 1e-8)
    source_rms = max(float(high_stats.get("source_rms", 0.0)), 1e-8)
    source_high *= float(np.clip(base_rms / source_rms, 0.18, 4.0))

    return {
        "base": torch.from_numpy(_exact_np(base, exact_samples)).float().view(1, 1, -1).to(engine.device),
        "source_high": torch.from_numpy(_exact_np(source_high, exact_samples)).float().view(1, 1, -1).to(engine.device),
        "target": torch.from_numpy(_exact_np(target_native, exact_samples)).float().view(1, 1, -1).to(engine.device),
        "source_f0": source_f0_warp.detach(),
        "target_f0": target_f0_t.detach(),
        "high_stats": high_stats,
    }


def _stft_mag(x, sr, n_fft=1024, hop=128):
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=window, return_complex=True,
    )
    mag = torch.log1p(spec.abs())
    freqs = torch.linspace(0.0, sr * 0.5, mag.shape[1], device=x.device, dtype=x.dtype)
    return mag, freqs


def highband_loss(pred, target, source_f0, target_f0, sr):
    p, freqs = _stft_mag(pred, sr)
    t, _ = _stft_mag(target, sr)
    hi = min(20000.0, sr * 0.5 - 100.0)
    mask = (freqs >= 7200.0) & (freqs <= hi)
    if not bool(mask.any()):
        return F.smooth_l1_loss(pred, target), {}
    pb = p[:, mask, :]
    tb = t[:, mask, :]
    spectral = F.smooth_l1_loss(pb, tb)

    if pb.shape[-1] > 1:
        flux = F.smooth_l1_loss(
            torch.relu(torch.diff(pb, dim=-1)),
            torch.relu(torch.diff(tb, dim=-1)),
        )
    else:
        flux = pred.new_tensor(0.0)

    if pb.shape[1] > 2:
        shape = F.smooth_l1_loss(torch.diff(pb, dim=1), torch.diff(tb, dim=1))
    else:
        shape = pred.new_tensor(0.0)

    bands = []
    for lo, band_hi in ((7200.0, 10000.0), (10000.0, 14000.0), (14000.0, hi)):
        m = (freqs >= lo) & (freqs < band_hi)
        if bool(m.any()):
            bands.append(F.smooth_l1_loss(p[:, m, :].mean(dim=1), t[:, m, :].mean(dim=1)))
    band_loss = torch.stack(bands).mean() if bands else pred.new_tensor(0.0)

    frames = p.shape[-1]
    sf0 = F.interpolate(source_f0, size=frames, mode="linear", align_corners=False).clamp_min(1.0)
    tf0 = F.interpolate(target_f0, size=frames, mode="linear", align_corners=False).clamp_min(1.0)
    freq_grid = freqs.view(1, -1, 1)
    source_order = torch.round(freq_grid / sf0)
    target_order = torch.round(freq_grid / tf0)
    source_distance = torch.abs(freq_grid - source_order * sf0) / sf0
    target_distance = torch.abs(freq_grid - target_order * tf0) / tf0
    source_harm = torch.exp(-0.5 * (source_distance / 0.13).pow(2))
    target_harm = torch.exp(-0.5 * (target_distance / 0.13).pow(2))
    leakage_region = (source_harm > 0.45) & (target_harm < 0.20) & mask.view(1, -1, 1)
    if bool(leakage_region.any()):
        excess = torch.relu((p - t) - 0.05)
        leak = excess[leakage_region.expand_as(excess)].pow(2).mean()
    else:
        leak = pred.new_tensor(0.0)

    loss = spectral + 0.42 * flux + 0.25 * shape + 0.30 * band_loss + 0.55 * leak
    return loss, {
        "spectral": float(spectral.detach()),
        "flux": float(flux.detach()),
        "shape": float(shape.detach()),
        "bands": float(band_loss.detach()),
        "source_f0_leak": float(leak.detach()),
    }


def evaluate(model, samples, sr):
    if not samples:
        return {}
    model.eval()
    rows = []
    with torch.inference_mode():
        for sample in samples:
            pred, residual, inject, suppress, _ = model(
                sample["base"], sample["source_high"], sample["source_f0"], sample["target_f0"]
            )
            base_loss, _ = highband_loss(sample["base"], sample["target"], sample["source_f0"], sample["target_f0"], sr)
            pred_loss, parts = highband_loss(pred, sample["target"], sample["source_f0"], sample["target_f0"], sr)
            rows.append({
                "base": float(base_loss.detach()),
                "pred": float(pred_loss.detach()),
                "inject": float(inject.mean().detach()),
                "suppress": float(suppress.mean().detach()),
                "residual_percent": 100.0 * float(torch.sqrt(torch.mean(residual.pow(2)) + 1e-12).cpu()) / max(float(torch.sqrt(torch.mean(sample['base'].pow(2)) + 1e-12).cpu()), 1e-8),
                **parts,
            })
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    base = mean("base")
    pred = mean("pred")
    return {
        "base_loss": base,
        "refined_loss": pred,
        "improvement_percent": 100.0 * (base - pred) / max(abs(base), 1e-8),
        "inject_mean": mean("inject"),
        "suppress_mean": mean("suppress"),
        "residual_percent": mean("residual_percent"),
        "source_f0_leak": mean("source_f0_leak"),
        "spectral": mean("spectral"),
        "flux": mean("flux"),
        "shape": mean("shape"),
        "bands": mean("bands"),
    }


def write_examples(model, samples, out_dir, sr, count=4):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        for i, sample in enumerate(samples[:count]):
            pred, residual, _, _, _ = model(
                sample["base"], sample["source_high"], sample["source_f0"], sample["target_f0"]
            )
            sf.write(out_dir / f"{i:02d}_base.wav", sample["base"][0, 0].cpu().numpy(), sr, subtype="FLOAT")
            sf.write(out_dir / f"{i:02d}_refined.wav", pred[0, 0].cpu().numpy(), sr, subtype="FLOAT")
            sf.write(out_dir / f"{i:02d}_target.wav", sample["target"][0, 0].cpu().numpy(), sr, subtype="FLOAT")
            sf.write(out_dir / f"{i:02d}_source_high.wav", sample["source_high"][0, 0].cpu().numpy(), sr, subtype="FLOAT")
            sf.write(out_dir / f"{i:02d}_residual.wav", residual[0, 0].cpu().numpy(), sr, subtype="FLOAT")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("voicebank")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--pairs", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=7e-4)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    voicebank = Path(args.voicebank).expanduser().resolve()
    config = load_config(root)
    engine = YuazDDSPResamplerEngine(
        config["yuaz_repo"], config["checkpoint"],
        transition_ms=config.get("transition_ms", 70),
        use_rvq=config.get("use_rvq", False),
        output_sr=config.get("output_sr", 44100),
        registry_path=config.get("registry_path"),
        ddsp_synthesis_sr=config.get("ddsp_synthesis_sr", 48000),
    )

    pair_specs, pair_report = build_multipitch_pairs(engine, voicebank, args.pairs)
    if len(pair_specs) < 4:
        raise RuntimeError(
            f"Only {len(pair_specs)} usable same-alias multipitch pairs were found; need at least 4."
        )
    print(json.dumps(pair_report, indent=2, ensure_ascii=False), flush=True)

    samples = []
    for i, pair in enumerate(pair_specs, 1):
        print(
            f"prepare [{i}/{len(pair_specs)}] {pair['alias']} "
            f"{pair['semitones']:.1f} st",
            flush=True,
        )
        try:
            sample = _prepare_base(engine, pair["source_entry"], pair["target_entry"])
            if sample is not None:
                sample["alias"] = pair["alias"]
                sample["semitones"] = float(pair["semitones"])
                samples.append(sample)
        except Exception as exc:
            print(f"skip: {exc}", flush=True)

    if len(samples) < 4:
        raise RuntimeError(f"Only {len(samples)} training samples could be prepared.")

    rng = random.Random(20260902)
    rng.shuffle(samples)
    val_count = max(2, min(8, len(samples) // 5))
    val = samples[:val_count]
    train = samples[val_count:]
    if not train:
        train = samples[val_count - 1:]
        val = samples[:val_count - 1]

    model = HighDetailRouter(sample_rate=engine.output_sr, hidden=32, frame_hop=128).to(engine.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=2e-5)

    out_dir = root / "high-detail-router-output"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    best_path = out_dir / "high_detail_router.pt"

    before = evaluate(model, val, engine.output_sr)
    print("before:", json.dumps(before, ensure_ascii=False), flush=True)

    for epoch in range(int(args.epochs)):
        model.train()
        rng.shuffle(train)
        running = []
        for idx, sample in enumerate(train, 1):
            optimizer.zero_grad(set_to_none=True)
            pred, residual, inject, suppress, _ = model(
                sample["base"], sample["source_high"], sample["source_f0"], sample["target_f0"]
            )
            loss, parts = highband_loss(
                pred, sample["target"], sample["source_f0"], sample["target_f0"], engine.output_sr
            )
            # Router should learn to use source detail, but not solve the task by
            # replacing the entire generated upper band.
            control_reg = 0.018 * inject.pow(2).mean() + 0.030 * suppress.pow(2).mean()
            residual_reg = 0.010 * torch.mean(residual.pow(2)) / (torch.mean(sample["base"].pow(2)) + 1e-7)
            total = loss + control_reg + residual_reg
            if not torch.isfinite(total):
                continue
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            running.append(float(total.detach()))
            if idx == 1 or idx % 8 == 0 or idx == len(train):
                print(
                    f"epoch {epoch + 1}/{args.epochs} [{idx}/{len(train)}] "
                    f"loss={float(total.detach()):.5f} "
                    f"leak={parts.get('source_f0_leak', 0.0):.5f}",
                    flush=True,
                )

        metrics = evaluate(model, val, engine.output_sr)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = float(np.mean(running)) if running else 0.0
        history.append(metrics)
        print("validation:", json.dumps(metrics, ensure_ascii=False), flush=True)
        if metrics.get("refined_loss", float("inf")) < best_loss:
            best_loss = metrics["refined_loss"]
            save_high_detail_router(best_path, model, {
                "training_format": TRAINING_FORMAT,
                "sample_rate": engine.output_sr,
                "hidden": model.hidden,
                "frame_hop": model.frame_hop,
                "voicebank": str(voicebank),
                "train_samples": len(train),
                "validation_samples": len(val),
                "best_epoch": epoch + 1,
                "best_validation_loss": best_loss,
                "pair_report": pair_report,
            })

    report = {
        "training_format": TRAINING_FORMAT,
        "voicebank": str(voicebank),
        "pair_report": pair_report,
        "prepared_samples": len(samples),
        "train_samples": len(train),
        "validation_samples": len(val),
        "before": before,
        "history": history,
        "best_model": str(best_path),
    }
    (out_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if best_path.is_file():
        payload = torch.load(best_path, map_location=engine.device, weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=False)
    write_examples(model, val, out_dir / "examples", engine.output_sr)
    print(f"Saved: {best_path}", flush=True)
    print(f"Examples: {out_dir / 'examples'}", flush=True)


if __name__ == "__main__":
    main()
