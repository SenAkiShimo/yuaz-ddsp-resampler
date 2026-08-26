#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from .clarity_refiner import ClarityRefiner, save_clarity_refiner
from .core import (
    YuazDDSPResamplerEngine,
    deterministic_decode,
    extract_detail_features,
    extract_f0,
    read_audio,
    stable_seed,
)
from .prepare import clarity_reconstruction_loss


def load_config(root):
    path = Path(root) / "config.json"
    if not path.exists():
        raise RuntimeError("Run configure-macos.command first.")
    return json.loads(path.read_text(encoding="utf-8"))


def list_wavs(root, limit):
    paths = sorted([p for p in Path(root).rglob("*.wav") if p.is_file()])
    if not paths:
        raise RuntimeError("No WAV files found in the selected folder.")
    rng = random.Random(20260826)
    rng.shuffle(paths)
    return paths[: min(int(limit), len(paths))]


def exact_length_np(x, n):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(n)
    if x.size < n:
        return np.pad(x, (0, n - x.size)).astype(np.float32)
    return x[:n].astype(np.float32)


def prepare_pair(engine, wav):
    audio = read_audio(wav, engine.sr)
    if len(audio) < int(0.12 * engine.sr):
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
    adapter, _, ai_controls, record = engine._models_for_input(wav)
    prototype_index = record.get("subbank_index") if record else None
    seed = stable_seed(str(wav), "clarity-refiner-train")
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
    if n < int(0.10 * engine.sr):
        return None
    target = torch.from_numpy(exact_length_np(audio, n)).float().view(1, 1, -1).to(engine.device)
    source = torch.from_numpy(exact_length_np(reconstructed, n)).float().view(1, 1, -1).to(engine.device)
    f0_use = F.interpolate(f0_aligned, size=max(1, int(round(n / engine.hop))), mode="linear", align_corners=False)
    voiced = np.flatnonzero(f0_np > 1.0)
    first_voiced_sample = int(round(int(voiced[0]) * engine.hop)) if voiced.size else 0
    articulation_end = min(n, first_voiced_sample + int(round(0.23 * engine.sr)))
    return {
        "path": str(wav),
        "target": target,
        "source": source,
        "detail": detail_t,
        "f0": f0_use,
        "first_voiced_sample": first_voiced_sample,
        "articulation_end_sample": articulation_end,
    }


def split_chunks(pair, sample_rate, chunk_ms=420):
    n = pair["source"].shape[-1]
    chunk = int(round(float(chunk_ms) * sample_rate / 1000.0))
    if n <= chunk:
        return [pair]
    fv = int(pair["first_voiced_sample"])
    starts = {
        max(0, min(n - chunk, fv - int(0.10 * sample_rate))),
        max(0, min(n - chunk, fv - int(0.03 * sample_rate))),
    }
    chunks = []
    for start in sorted(starts):
        end = start + chunk
        q = dict(pair)
        q["source"] = pair["source"][..., start:end]
        q["target"] = pair["target"][..., start:end]
        q["first_voiced_sample"] = max(0, fv - start)
        q["articulation_end_sample"] = max(1, min(chunk, int(pair["articulation_end_sample"]) - start))
        chunks.append(q)
    return chunks


def validate(model, pairs, sr):
    if not pairs:
        return None
    losses = []
    model.eval()
    with torch.inference_mode():
        for pair in pairs:
            pred, _, _ = model(pair["source"], pair["detail"], pair["f0"])
            loss, _ = clarity_reconstruction_loss(
                pair["target"], pred, sr,
                pair["first_voiced_sample"],
                pair_mode=False,
                articulation_end_sample=pair["articulation_end_sample"],
            )
            losses.append(float(loss))
    return float(np.mean(losses)) if losses else None


def write_examples(model, pairs, out_dir, sr):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        for i, pair in enumerate(pairs[:4]):
            pred, residual, gate = model(pair["source"], pair["detail"], pair["f0"])
            prefix = f"{i:02d}"
            sf.write(out_dir / f"{prefix}_target.wav", pair["target"][0, 0].cpu().numpy(), sr, subtype="PCM_16")
            sf.write(out_dir / f"{prefix}_yuaz.wav", pair["source"][0, 0].cpu().numpy(), sr, subtype="PCM_16")
            sf.write(out_dir / f"{prefix}_clarity.wav", pred[0, 0].cpu().numpy(), sr, subtype="PCM_16")
            sf.write(out_dir / f"{prefix}_residual.wav", residual[0, 0].cpu().numpy(), sr, subtype="FLOAT")
            np.save(out_dir / f"{prefix}_gate.npy", gate[0, 0].cpu().numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("voicebank")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--limit", type=int, default=96)
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

    wavs = list_wavs(voicebank, args.limit)
    pairs = []
    print(f"Preparing up to {len(wavs)} same-pitch reconstruction pairs...")
    for i, wav in enumerate(wavs, 1):
        try:
            pair = prepare_pair(engine, wav)
        except Exception as exc:
            print(f"skip {wav.name}: {exc}")
            continue
        if pair is None:
            continue
        pairs.extend(split_chunks(pair, engine.sr))
        print(f"[{i}/{len(wavs)}] {wav.name}")

    if len(pairs) < 8:
        raise RuntimeError(f"Too few usable training pairs: {len(pairs)}")

    rng = random.Random(20260826)
    rng.shuffle(pairs)
    val_count = max(2, min(12, len(pairs) // 8))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    model = ClarityRefiner(sample_rate=engine.sr).to(engine.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-5)
    best_loss = math.inf
    best_state = None
    history = []

    for epoch in range(1, int(args.epochs) + 1):
        rng.shuffle(train_pairs)
        model.train()
        running = []
        for pair in train_pairs:
            optimizer.zero_grad(set_to_none=True)
            pred, residual, gate = model(pair["source"], pair["detail"], pair["f0"])
            loss, parts = clarity_reconstruction_loss(
                pair["target"], pred, engine.sr,
                pair["first_voiced_sample"],
                pair_mode=False,
                articulation_end_sample=pair["articulation_end_sample"],
            )
            residual_penalty = torch.mean(torch.abs(residual))
            stable_penalty = torch.mean(torch.abs(residual) * (1.0 - gate))
            total = loss + 0.08 * residual_penalty + 0.30 * stable_penalty
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running.append(float(total.detach()))

        val = validate(model, val_pairs, engine.sr)
        train_loss = float(np.mean(running)) if running else math.inf
        score = val if val is not None else train_loss
        history.append({"epoch": epoch, "train": train_loss, "val": val})
        print(f"epoch {epoch}: train={train_loss:.6f} val={val if val is not None else 'n/a'}")
        if score < best_loss:
            best_loss = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out = root / "clarity-refiner-output"
    out.mkdir(exist_ok=True)
    model_path = out / "clarity_refiner_test.pt"
    save_clarity_refiner(model_path, model, metadata={
        "sample_rate": engine.sr,
        "training_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "voicebank": str(voicebank),
        "history": history,
        "same_pitch_only": True,
        "runtime_integrated": False,
    })
    write_examples(model, val_pairs, out / "examples", engine.sr)
    (out / "training.json").write_text(json.dumps({
        "history": history,
        "best_loss": best_loss,
        "train_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "model": str(model_path),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {model_path}")
    print(f"A/B examples: {out / 'examples'}")


if __name__ == "__main__":
    main()
