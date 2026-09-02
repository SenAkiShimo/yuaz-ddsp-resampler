#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from .core import YuazDDSPResamplerEngine
from .high_detail_tf_generator import HighDetailTFGenerator, save_high_detail_tf
from . import train_high_detail_router_entry as entry
from . import train_high_detail_router_v3 as v3


TRAINING_FORMAT = 1
base = entry.base


def load_config(root):
    path = Path(root) / "config.json"
    if not path.exists():
        raise RuntimeError("Run configure-macos.command first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _stft(x, sr, n_fft=1024, hop=128):
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=window, return_complex=True,
    )
    freqs = torch.linspace(0.0, sr * 0.5, spec.shape[1], device=x.device, dtype=x.dtype)
    return spec, freqs


def tf_loss(model, sample, sr):
    pred, residual, mask, _, _ = model(
        sample["base"], sample["source_high"], sample["source_f0"], sample["target_f0"]
    )
    r_spec, freqs = _stft(residual, sr, model.n_fft, model.hop)
    t_spec, _ = _stft(sample["target_detail"], sr, model.n_fft, model.hop)
    frames = min(r_spec.shape[-1], t_spec.shape[-1])
    r_spec = r_spec[..., :frames]
    t_spec = t_spec[..., :frames]
    mask = mask[..., :frames]

    hi = min(20000.0, sr * 0.5 - 100.0)
    band = (freqs >= 7200.0) & (freqs <= hi)
    rb = torch.log1p(42.0 * r_spec[:, band, :].abs())
    tb = torch.log1p(42.0 * t_spec[:, band, :].abs())

    delta = rb - tb
    missing = torch.relu(-delta)
    extra = torch.relu(delta)
    spectral = (2.2 * missing.pow(2) + 0.72 * extra.pow(2)).mean()

    if rb.shape[-1] > 1:
        rflux = torch.relu(torch.diff(rb, dim=-1))
        tflux = torch.relu(torch.diff(tb, dim=-1))
        flux = F.smooth_l1_loss(rflux, tflux)
    else:
        flux = spectral.new_tensor(0.0)

    if rb.shape[1] > 2:
        shape = F.smooth_l1_loss(torch.diff(rb, dim=1), torch.diff(tb, dim=1))
    else:
        shape = spectral.new_tensor(0.0)

    sf0 = F.interpolate(sample["source_f0"], size=frames, mode="linear", align_corners=False).clamp_min(1.0)
    tf0 = F.interpolate(sample["target_f0"], size=frames, mode="linear", align_corners=False).clamp_min(1.0)
    grid = freqs.view(1, -1, 1)
    so = torch.round(grid / sf0)
    to = torch.round(grid / tf0)
    sd = torch.abs(grid - so * sf0) / sf0
    td = torch.abs(grid - to * tf0) / tf0
    sh = torch.exp(-0.5 * (sd / 0.13).pow(2))
    th = torch.exp(-0.5 * (td / 0.13).pow(2))
    leak_region = (sh > 0.50) & (th < 0.18) & band.view(1, -1, 1)
    if bool(leak_region.any()):
        leak = (r_spec.abs() * leak_region.to(r_spec.real.dtype)).pow(2).sum() / (
            leak_region.sum() + 1e-8
        )
    else:
        leak = spectral.new_tensor(0.0)

    # Avoid the old collapse-to-zero route while still allowing selective rejection.
    high_mask = mask[:, band, :]
    mask_floor = torch.relu(0.10 - high_mask).pow(2).mean()
    mask_tv = torch.abs(torch.diff(high_mask, dim=-1)).mean() if high_mask.shape[-1] > 1 else spectral.new_tensor(0.0)

    total = spectral + 0.34 * flux + 0.20 * shape + 0.35 * leak + 0.12 * mask_floor + 0.006 * mask_tv
    parts = {
        "spectral": float(spectral.detach()),
        "flux": float(flux.detach()),
        "shape": float(shape.detach()),
        "source_f0_leak": float(leak.detach()),
        "mask_mean": float(high_mask.mean().detach()),
        "mask_active": float((high_mask > 0.20).to(high_mask.dtype).mean().detach()),
    }
    return total, pred, residual, mask, parts


def evaluate(model, samples, sr):
    rows = []
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            loss, _, residual, mask, parts = tf_loss(model, sample, sr)
            target_rms = torch.sqrt(torch.mean(sample["target_detail"].pow(2)) + 1e-12)
            residual_rms = torch.sqrt(torch.mean(residual.pow(2)) + 1e-12)
            rows.append({
                "loss": float(loss),
                "detail_rms_ratio": float(residual_rms / (target_rms + 1e-8)),
                **parts,
            })
    if not rows:
        return {}
    mean = lambda key: float(np.mean([x[key] for x in rows]))
    return {
        "loss": mean("loss"),
        "spectral": mean("spectral"),
        "flux": mean("flux"),
        "shape": mean("shape"),
        "source_f0_leak": mean("source_f0_leak"),
        "mask_mean": mean("mask_mean"),
        "mask_active": mean("mask_active"),
        "detail_rms_ratio": mean("detail_rms_ratio"),
    }


def write_examples(model, samples, out_dir, sr, count=4):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        for i, sample in enumerate(samples[:count]):
            _, pred, residual, _, _ = tf_loss(model, sample, sr)
            sf.write(out_dir / f"{i:02d}_base.wav", sample["base"][0, 0].cpu().numpy(), sr, subtype="FLOAT")
            sf.write(out_dir / f"{i:02d}_refined.wav", pred[0, 0].cpu().numpy(), sr, subtype="FLOAT")
            sf.write(out_dir / f"{i:02d}_teacher.wav", sample["teacher"][0, 0].cpu().numpy(), sr, subtype="FLOAT")
            sf.write(out_dir / f"{i:02d}_source_high.wav", sample["source_high"][0, 0].cpu().numpy(), sr, subtype="FLOAT")
            sf.write(out_dir / f"{i:02d}_target_high.wav", sample["target_detail"][0, 0].cpu().numpy(), sr, subtype="FLOAT")
            sf.write(out_dir / f"{i:02d}_predicted_high.wav", residual[0, 0].cpu().numpy(), sr, subtype="FLOAT")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("voicebank")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--pairs", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=4e-4)
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

    pair_specs, pair_report = entry.build_multipitch_pairs_fast(engine, voicebank, args.pairs)
    print(json.dumps(pair_report, indent=2, ensure_ascii=False), flush=True)

    samples = []
    for i, pair in enumerate(pair_specs, 1):
        print(f"prepare [{i}/{len(pair_specs)}] {pair['alias']} {pair['semitones']:.1f} st", flush=True)
        try:
            sample = v3._prepare_v3(engine, pair["source_entry"], pair["target_entry"])
            if sample is not None:
                sample["alias"] = pair["alias"]
                samples.append(sample)
        except Exception as exc:
            print(f"skip: {exc}", flush=True)

    if len(samples) < 12:
        raise RuntimeError(f"Only {len(samples)} TF training samples could be prepared.")

    rng = random.Random(20260902)
    rng.shuffle(samples)
    val_count = max(6, min(16, len(samples) // 5))
    val = samples[:val_count]
    train = samples[val_count:]

    model = HighDetailTFGenerator(sample_rate=engine.output_sr, n_fft=1024, hop=128, hidden=24).to(engine.device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=2e-5)

    out_dir = root / "high-detail-tf-output"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "high_detail_tf.pt"
    best_loss = float("inf")
    history = []

    before = evaluate(model, val, engine.output_sr)
    print("before:", json.dumps(before, ensure_ascii=False), flush=True)

    for epoch in range(int(args.epochs)):
        model.train()
        rng.shuffle(train)
        running = []
        for idx, sample in enumerate(train, 1):
            opt.zero_grad(set_to_none=True)
            loss, _, _, _, parts = tf_loss(model, sample, engine.output_sr)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            opt.step()
            running.append(float(loss.detach()))
            if idx == 1 or idx % 12 == 0 or idx == len(train):
                print(
                    f"epoch {epoch + 1}/{args.epochs} [{idx}/{len(train)}] "
                    f"loss={float(loss.detach()):.5f} mask={parts['mask_mean']:.3f} "
                    f"leak={parts['source_f0_leak']:.5f}",
                    flush=True,
                )

        metrics = evaluate(model, val, engine.output_sr)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = float(np.mean(running)) if running else 0.0
        history.append(metrics)
        print("validation:", json.dumps(metrics, ensure_ascii=False), flush=True)
        if metrics.get("loss", float("inf")) < best_loss:
            best_loss = metrics["loss"]
            save_high_detail_tf(best_path, model, {
                "training_format": TRAINING_FORMAT,
                "sample_rate": engine.output_sr,
                "n_fft": model.n_fft,
                "hop": model.hop,
                "hidden": model.hidden,
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
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if best_path.is_file():
        payload = torch.load(best_path, map_location=engine.device, weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
    write_examples(model, val, out_dir / "examples", engine.output_sr)
    print(f"Saved: {best_path}", flush=True)
    print(f"Examples: {out_dir / 'examples'}", flush=True)


if __name__ == "__main__":
    main()
