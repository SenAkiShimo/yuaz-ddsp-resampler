#!/usr/bin/env python3
import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from .clarity_refiner import ClarityRefiner, save_clarity_refiner
from .core import (
    YuazDDSPResamplerEngine,
    crop_oto,
    deterministic_decode,
    extract_detail_features,
    extract_f0,
    read_audio,
    stable_seed,
)
from .prepare import clarity_reconstruction_loss
from .voicebank import scan_voicebank


TEACHER_FORMAT = 1


def load_config(root):
    path = Path(root) / "config.json"
    if not path.exists():
        raise RuntimeError("Run configure-macos.command first.")
    return json.loads(path.read_text(encoding="utf-8"))


def articulation_rich(entry):
    alias = str(entry.alias or "").strip()
    tokens = [x for x in re.split(r"[\s_-]+", alias) if x]
    return bool(
        float(entry.consonant) >= 35.0
        or float(entry.preutterance) >= 25.0
        or len(tokens) >= 2
    )


def list_oto_entries(root, limit):
    scan = scan_voicebank(root)
    entries = list(scan["entries"])
    if not entries:
        raise RuntimeError("No usable OTO entries found in the selected voicebank.")

    rng = random.Random(20260826)
    rich = [x for x in entries if articulation_rich(x)]
    simple = [x for x in entries if not articulation_rich(x)]
    rng.shuffle(rich)
    rng.shuffle(simple)

    limit = min(int(limit), len(entries))
    rich_target = min(len(rich), int(round(limit * 0.70)))
    selected = rich[:rich_target]
    remaining = limit - len(selected)
    selected.extend(simple[:remaining])
    remaining = limit - len(selected)
    if remaining > 0:
        selected.extend(rich[rich_target:rich_target + remaining])
    rng.shuffle(selected)
    return selected, {
        "oto_files": len(scan["oto_files"]),
        "oto_entries": len(entries),
        "articulation_rich_entries": len(rich),
        "selected": len(selected),
        "selected_articulation_rich": sum(1 for x in selected if articulation_rich(x)),
    }


def exact_length_np(x, n):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(n)
    if x.size < n:
        return np.pad(x, (0, n - x.size)).astype(np.float32)
    return x[:n].astype(np.float32)


def prepare_pair(engine, entry):
    wav = Path(entry.wav_path)
    audio = read_audio(wav, engine.sr)
    audio = crop_oto(audio, engine.sr, entry.offset, entry.cutoff)
    if len(audio) < int(0.10 * engine.sr):
        return None

    f0_np = extract_f0(audio, engine.sr, engine.hop)
    if not np.any(f0_np > 1.0):
        return None

    audio_t = torch.from_numpy(audio).float().view(1, 1, -1).to(engine.device)
    f0_t = torch.from_numpy(f0_np).float().view(1, 1, -1).to(engine.device)
    detail_np = extract_detail_features(audio, engine.sr, engine.hop)
    detail_t = torch.from_numpy(detail_np).float().unsqueeze(0).to(engine.device)

    with torch.inference_mode():
        z, f0_aligned = engine.encoder(audio_t, f0_override=f0_t)

    adapter, _, _, record = engine._models_for_input(wav)
    prototype_index = record.get("subbank_index") if record else None
    seed = stable_seed(
        str(wav), str(entry.alias), float(entry.offset), float(entry.cutoff),
        "clarity-refiner-train-v3",
    )
    reconstructed = deterministic_decode(
        engine.decoder,
        f0_aligned,
        z,
        seed,
        adapter=adapter,
        detail=detail_t,
        prototype_index=prototype_index,
    )

    n = min(len(audio), len(reconstructed))
    if n < int(0.09 * engine.sr):
        return None

    target = torch.from_numpy(exact_length_np(audio, n)).float().view(1, 1, -1).to(engine.device)
    source = torch.from_numpy(exact_length_np(reconstructed, n)).float().view(1, 1, -1).to(engine.device)
    f0_use = F.interpolate(
        f0_aligned,
        size=max(1, int(round(n / engine.hop))),
        mode="linear",
        align_corners=False,
    )

    voiced = np.flatnonzero(f0_np > 1.0)
    first_voiced_sample = int(round(int(voiced[0]) * engine.hop)) if voiced.size else 0
    oto_consonant_end = int(round(max(0.0, float(entry.consonant)) * engine.sr / 1000.0))
    articulation_end = min(
        n,
        max(
            first_voiced_sample + int(round(0.16 * engine.sr)),
            oto_consonant_end + int(round(0.08 * engine.sr)),
        ),
    )

    return {
        "path": str(wav),
        "alias": str(entry.alias),
        "offset": float(entry.offset),
        "cutoff": float(entry.cutoff),
        "consonant": float(entry.consonant),
        "preutterance": float(entry.preutterance),
        "articulation_rich": bool(articulation_rich(entry)),
        "target": target,
        "source": source,
        "detail": detail_t,
        "f0": f0_use,
        "first_voiced_sample": first_voiced_sample,
        "articulation_end_sample": articulation_end,
    }


def _smoothstep(x):
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def articulation_sample_gate(pair, sr):
    n = int(pair["source"].shape[-1])
    gate = pair["source"].new_zeros((1, 1, n))
    fv = int(pair["first_voiced_sample"])
    end = int(pair["articulation_end_sample"])
    start = max(0, fv - int(round(0.10 * sr)))
    attack = min(int(round(0.012 * sr)), max(0, end - start))
    if end > start:
        gate[..., start:end] = 1.0
    if attack > 1 and start > 0:
        a0 = max(0, start - attack)
        u = torch.linspace(0.0, 1.0, start - a0, device=gate.device, dtype=gate.dtype)
        gate[..., a0:start] = 0.5 - 0.5 * torch.cos(torch.pi * u)
    decay_end = min(n, end + int(round(0.09 * sr)))
    if decay_end > end:
        u = torch.linspace(0.0, 1.0, decay_end - end, device=gate.device, dtype=gate.dtype)
        gate[..., end:decay_end] = 0.5 + 0.5 * torch.cos(torch.pi * u)
    return gate


def build_phase_compatible_teacher(pair, sr):
    source = pair["source"].squeeze(1)
    target = pair["target"].squeeze(1)
    n = min(source.shape[-1], target.shape[-1])
    source = source[..., :n]
    target = target[..., :n]

    n_fft = 512
    hop = 64
    window = torch.hann_window(n_fft, device=source.device, dtype=source.dtype)
    source_spec = torch.stft(
        source, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=window, return_complex=True,
    )
    target_spec = torch.stft(
        target, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=window, return_complex=True,
    )

    eps = 1e-5
    source_mag = source_spec.abs()
    target_mag = target_spec.abs()
    log_ratio = torch.log(target_mag + eps) - torch.log(source_mag + eps)
    log_ratio = log_ratio - torch.median(log_ratio, dim=1, keepdim=True).values
    max_log = float(8.0 * math.log(10.0) / 20.0)
    log_ratio = torch.clamp(log_ratio, -max_log, max_log)

    bins = source_spec.shape[1]
    frames = source_spec.shape[2]
    freqs = torch.linspace(0.0, sr * 0.5, bins, device=source.device, dtype=source.dtype).view(1, bins, 1)
    frequency_gate = _smoothstep((freqs - 1200.0) / 1800.0)

    sample_gate = articulation_sample_gate(pair, sr)
    time_gate = F.interpolate(sample_gate, size=frames, mode="linear", align_corners=False)

    f0 = pair["f0"]
    if f0.dim() == 2:
        f0 = f0.unsqueeze(1)
    f0_frames = F.interpolate(f0, size=frames, mode="linear", align_corners=False)
    voiced = (f0_frames > 1.0).to(source.dtype)
    safe_f0 = torch.clamp(f0_frames, min=40.0)
    harmonic_index = torch.round(freqs / safe_f0)
    harmonic_distance = torch.abs(freqs - harmonic_index * safe_f0) / safe_f0
    harmonic_peak = torch.exp(-0.5 * torch.pow(harmonic_distance / 0.11, 2.0))
    harmonic_peak = harmonic_peak * (harmonic_index >= 1.0).to(source.dtype) * voiced
    nonperiodic_gate = 1.0 - 0.90 * harmonic_peak

    transfer = frequency_gate * time_gate * nonperiodic_gate
    desired_mag = torch.exp(torch.log(source_mag + eps) + log_ratio * transfer)
    source_phase = source_spec / (source_mag + eps)
    teacher_spec = desired_mag * source_phase
    teacher = torch.istft(
        teacher_spec, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=window, length=n,
    ).unsqueeze(1)

    residual = teacher - pair["source"][..., :n]
    weight = torch.clamp(sample_gate[..., :n], min=0.05)
    source_rms = torch.sqrt(
        torch.sum(pair["source"][..., :n].pow(2) * weight) / (torch.sum(weight) + 1e-8) + 1e-8
    )
    residual_rms = torch.sqrt(
        torch.sum(residual.pow(2) * weight) / (torch.sum(weight) + 1e-8) + 1e-8
    )
    limit = 0.10 * source_rms + 1e-7
    scale = torch.clamp(limit / (residual_rms + 1e-8), max=1.0)
    residual = residual * scale
    teacher = torch.clamp(pair["source"][..., :n] + residual, -1.2, 1.2)
    return teacher.detach(), residual.detach(), sample_gate[..., :n].detach(), {
        "teacher_scale": float(scale.detach().cpu()),
        "transfer_mean": float(transfer.mean().detach().cpu()),
    }


def clarity_loss(pair, pred, sr):
    return clarity_reconstruction_loss(
        pair["target"].squeeze(1),
        pred.squeeze(1),
        sr,
        pair["first_voiced_sample"],
        pair_mode=False,
        articulation_end_sample=pair["articulation_end_sample"],
    )


def teacher_clarity_loss(pair, teacher, pred, sr):
    return clarity_reconstruction_loss(
        teacher.squeeze(1),
        pred.squeeze(1),
        sr,
        pair["first_voiced_sample"],
        pair_mode=False,
        articulation_end_sample=pair["articulation_end_sample"],
    )


def rms(x):
    return float(torch.sqrt(torch.mean(x.detach().pow(2)) + 1e-12).cpu())


def weighted_rms(x, weight):
    x = x.detach()
    weight = weight.detach()
    num = torch.sum(x.pow(2) * weight)
    den = torch.sum(weight) + 1e-8
    return float(torch.sqrt(num / den + 1e-12).cpu())


def normalized_teacher_error(source, residual, teacher_residual):
    scale = torch.sqrt(torch.mean(source.detach().pow(2)) + 1e-8).detach()
    return F.l1_loss(residual / scale, teacher_residual / scale)


def residual_cosine(a, b):
    a = a.detach().reshape(-1)
    b = b.detach().reshape(-1)
    den = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) + 1e-12
    return float((torch.dot(a, b) / den).cpu())


def diagnose(model, pairs, sr):
    if not pairs:
        return None

    rows = []
    model.eval()
    with torch.inference_mode():
        for pair in pairs:
            teacher, teacher_residual, teacher_gate, teacher_stats = build_phase_compatible_teacher(pair, sr)
            base_loss, _ = clarity_loss(pair, pair["source"], sr)
            teacher_base_loss, _ = teacher_clarity_loss(pair, teacher, pair["source"], sr)
            pred, residual, gate = model(
                pair["source"], pair["detail"], pair["f0"],
                articulation_mask=teacher_gate,
            )
            refined_loss, _ = clarity_loss(pair, pred, sr)
            teacher_refined_loss, _ = teacher_clarity_loss(pair, teacher, pred, sr)
            teacher_error = normalized_teacher_error(pair["source"], residual, teacher_residual)
            source_rms = rms(pair["source"])
            residual_rms = rms(residual)
            teacher_rms = rms(teacher_residual)
            gate_mean = float(torch.mean(gate).cpu())
            gate_coverage = float(torch.mean((gate > 0.10).to(gate.dtype)).cpu())
            onset_rms = weighted_rms(residual, torch.clamp(gate, 0.0, 1.0))
            stable_rms = weighted_rms(residual, torch.clamp(1.0 - gate, 0.0, 1.0))
            rows.append({
                "base_loss": float(base_loss),
                "refined_loss": float(refined_loss),
                "teacher_base_loss": float(teacher_base_loss),
                "teacher_refined_loss": float(teacher_refined_loss),
                "teacher_error": float(teacher_error),
                "source_rms": source_rms,
                "residual_rms": residual_rms,
                "residual_percent": 100.0 * residual_rms / max(source_rms, 1e-8),
                "teacher_residual_rms": teacher_rms,
                "teacher_residual_percent": 100.0 * teacher_rms / max(source_rms, 1e-8),
                "teacher_cosine": residual_cosine(residual, teacher_residual),
                "gate_mean": gate_mean,
                "gate_coverage": gate_coverage,
                "onset_residual_rms": onset_rms,
                "stable_residual_rms": stable_rms,
                "teacher_scale": teacher_stats["teacher_scale"],
            })

    def mean(key):
        return float(np.mean([x[key] for x in rows])) if rows else 0.0

    base = mean("base_loss")
    refined = mean("refined_loss")
    teacher_base = mean("teacher_base_loss")
    teacher_refined = mean("teacher_refined_loss")
    output_weight_rms = float(model.output.weight.detach().pow(2).mean().sqrt().cpu())
    return {
        "base_loss": base,
        "refined_loss": refined,
        "improvement_percent": 100.0 * (base - refined) / max(abs(base), 1e-8),
        "teacher_base_loss": teacher_base,
        "teacher_refined_loss": teacher_refined,
        "teacher_improvement_percent": 100.0 * (teacher_base - teacher_refined) / max(abs(teacher_base), 1e-8),
        "teacher_error": mean("teacher_error"),
        "residual_rms": mean("residual_rms"),
        "residual_percent": mean("residual_percent"),
        "teacher_residual_rms": mean("teacher_residual_rms"),
        "teacher_residual_percent": mean("teacher_residual_percent"),
        "teacher_cosine": mean("teacher_cosine"),
        "teacher_scale": mean("teacher_scale"),
        "gate_mean": mean("gate_mean"),
        "gate_coverage_percent": 100.0 * mean("gate_coverage"),
        "onset_residual_rms": mean("onset_residual_rms"),
        "stable_residual_rms": mean("stable_residual_rms"),
        "output_weight_rms": output_weight_rms,
        "samples": len(rows),
    }


def print_diagnostics(label, d):
    if not d:
        return
    ratio = d["onset_residual_rms"] / max(d["stable_residual_rms"], 1e-9)
    print(
        f"{label}: original base={d['base_loss']:.6f} refined={d['refined_loss']:.6f} "
        f"improve={d['improvement_percent']:+.2f}%"
    )
    print(
        f"  teacher base={d['teacher_base_loss']:.6f} refined={d['teacher_refined_loss']:.6f} "
        f"improve={d['teacher_improvement_percent']:+.2f}% error={d['teacher_error']:.6f}"
    )
    print(
        f"  residual/source={d['residual_percent']:.3f}% "
        f"teacher_residual/source={d['teacher_residual_percent']:.3f}% "
        f"teacher_cosine={d['teacher_cosine']:.3f} teacher_scale={d['teacher_scale']:.3f}"
    )
    print(
        f"  gate_coverage={d['gate_coverage_percent']:.1f}% "
        f"onset_residual_rms={d['onset_residual_rms']:.8f} "
        f"stable_residual_rms={d['stable_residual_rms']:.8f} "
        f"onset/stable={ratio:.2f}x output_weight_rms={d['output_weight_rms']:.8f}"
    )


def write_examples(model, pairs, out_dir, sr):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    manifest = []
    with torch.inference_mode():
        for i, pair in enumerate(pairs[:6]):
            teacher, teacher_residual, teacher_gate, _ = build_phase_compatible_teacher(pair, sr)
            pred, residual, gate = model(
                pair["source"], pair["detail"], pair["f0"],
                articulation_mask=teacher_gate,
            )
            prefix = f"{i:02d}"
            sf.write(out_dir / f"{prefix}_target.wav", pair["target"][0, 0].cpu().numpy(), sr, subtype="PCM_16")
            sf.write(out_dir / f"{prefix}_yuaz.wav", pair["source"][0, 0].cpu().numpy(), sr, subtype="PCM_16")
            sf.write(out_dir / f"{prefix}_teacher.wav", teacher[0, 0].cpu().numpy(), sr, subtype="PCM_16")
            sf.write(out_dir / f"{prefix}_clarity.wav", pred[0, 0].cpu().numpy(), sr, subtype="PCM_16")
            sf.write(out_dir / f"{prefix}_teacher_residual.wav", teacher_residual[0, 0].cpu().numpy(), sr, subtype="FLOAT")
            sf.write(out_dir / f"{prefix}_residual.wav", residual[0, 0].cpu().numpy(), sr, subtype="FLOAT")
            np.save(out_dir / f"{prefix}_gate.npy", gate[0, 0].cpu().numpy())
            manifest.append({
                "prefix": prefix,
                "alias": pair["alias"],
                "wav": pair["path"],
                "offset": pair["offset"],
                "cutoff": pair["cutoff"],
                "consonant": pair["consonant"],
                "preutterance": pair["preutterance"],
                "articulation_rich": pair["articulation_rich"],
            })
    (out_dir / "examples.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("voicebank")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-4)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    voicebank = Path(args.voicebank).expanduser().resolve()
    config = load_config(root)
    engine = YuazDDSPResamplerEngine(
        config["yuaz_repo"], config["checkpoint"],
        transition_ms=config.get("transition_ms", 70),
        use_rvq=config.get("use_rvq", False),
        output_sr=config.get("output_sr", 44100),
        registry_path=config.get("registry_path"),
    )

    entries, data_summary = list_oto_entries(voicebank, args.limit)
    print(
        f"OTO dataset: {data_summary['oto_entries']} entries from {data_summary['oto_files']} oto.ini; "
        f"selected {data_summary['selected']} ({data_summary['selected_articulation_rich']} articulation-rich)."
    )
    print(f"Preparing up to {len(entries)} OTO-defined same-pitch reconstruction pairs...")

    pairs = []
    for i, entry in enumerate(entries, 1):
        try:
            pair = prepare_pair(engine, entry)
        except Exception as exc:
            print(f"skip {entry.alias} <- {Path(entry.wav_path).name}: {exc}")
            continue
        if pair is None:
            continue
        pairs.append(pair)
        print(f"[{i}/{len(entries)}] {entry.alias} <- {Path(entry.wav_path).name}")

    if len(pairs) < 8:
        raise RuntimeError(f"Too few usable training pairs: {len(pairs)}")

    rng = random.Random(20260826)
    rng.shuffle(pairs)
    val_count = max(4, min(12, len(pairs) // 6))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    model = ClarityRefiner(sample_rate=engine.sr).to(engine.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-5)
    best_score = math.inf
    best_state = None
    history = []

    initial_diag = diagnose(model, val_pairs, engine.sr)
    print_diagnostics("before training", initial_diag)

    for epoch in range(1, int(args.epochs) + 1):
        rng.shuffle(train_pairs)
        model.train()
        running = []
        for pair in train_pairs:
            teacher, teacher_residual, teacher_gate, _ = build_phase_compatible_teacher(pair, engine.sr)
            optimizer.zero_grad(set_to_none=True)
            pred, residual, gate = model(
                pair["source"], pair["detail"], pair["f0"],
                articulation_mask=teacher_gate,
            )
            original_loss, _ = clarity_loss(pair, pred, engine.sr)
            teacher_loss, _ = teacher_clarity_loss(pair, teacher, pred, engine.sr)
            teacher_error = normalized_teacher_error(pair["source"], residual, teacher_residual)
            stable_penalty = torch.mean(torch.abs(residual) * (1.0 - gate))
            total = teacher_error + 0.35 * teacher_loss + 0.08 * original_loss + 0.12 * stable_penalty
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running.append(float(total.detach()))

        train_loss = float(np.mean(running)) if running else math.inf
        diag = diagnose(model, val_pairs, engine.sr)
        score = (diag["teacher_error"] + 0.25 * diag["teacher_refined_loss"]) if diag else train_loss
        history.append({"epoch": epoch, "train": train_loss, "validation": diag})
        print(f"epoch {epoch}: train={train_loss:.6f}")
        print_diagnostics("  validation", diag)
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    final_diag = diagnose(model, val_pairs, engine.sr)
    print_diagnostics("best checkpoint", final_diag)

    out = root / "clarity-refiner-output"
    out.mkdir(exist_ok=True)
    model_path = out / "clarity_refiner_test.pt"
    save_clarity_refiner(model_path, model, metadata={
        "sample_rate": engine.sr,
        "training_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "voicebank": str(voicebank),
        "history": history,
        "initial_diagnostics": initial_diag,
        "final_diagnostics": final_diag,
        "data_summary": data_summary,
        "same_pitch_only": True,
        "oto_defined_units": True,
        "teacher_format": TEACHER_FORMAT,
        "teacher_mode": "source-phase/nonperiodic-articulation-logmag",
        "runtime_integrated": False,
    })
    write_examples(model, val_pairs, out / "examples", engine.sr)
    report = {
        "history": history,
        "best_score": best_score,
        "train_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "model": str(model_path),
        "data_summary": data_summary,
        "teacher_format": TEACHER_FORMAT,
        "initial_diagnostics": initial_diag,
        "final_diagnostics": final_diag,
    }
    (out / "training.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved: {model_path}")
    print(f"Report: {out / 'training.json'}")
    print(f"A/B examples: {out / 'examples'}")


if __name__ == "__main__":
    main()
