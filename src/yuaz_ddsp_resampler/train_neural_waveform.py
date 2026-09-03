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

from .core import YuazDDSPResamplerEngine, crop_oto, stable_seed
from .neural_waveform import (
    YuazNeuralWaveformDecoder,
    build_neural_conditioning,
    save_neural_waveform_decoder,
)


SAMPLE_RATE = 48000
PITCH_BUCKETS = (
    ("near", 0.0, 3.0),
    ("medium", 3.0, 7.0),
    ("far", 7.0, 12.0),
    ("extreme", 12.0, 1e9),
)


def exact_length(x, n):
    n = int(n)
    if x.shape[-1] == n:
        return x
    if x.shape[-1] > n:
        return x[..., :n]
    return F.pad(x, (0, n - x.shape[-1]))


def load_cache(path, device):
    with np.load(path, allow_pickle=False) as data:
        return {
            "latent": torch.from_numpy(data["latent"].astype(np.float32)).unsqueeze(0).to(device),
            "f0": torch.from_numpy(data["f0"].astype(np.float32)).view(1, 1, -1).to(device),
            "detail": torch.from_numpy(data["detail"].astype(np.float32)).unsqueeze(0).to(device),
        }


def read_fullband_target(voicebank_root, item, samples):
    wav = Path(voicebank_root) / str(item["relative_wav"])
    audio, sr = sf.read(wav, always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    audio = crop_oto(audio, int(sr), float(item.get("offset", 0.0)), float(item.get("cutoff", 0.0)))
    if int(sr) != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=int(sr), target_sr=SAMPLE_RATE).astype(np.float32)
    if len(audio) < int(samples):
        audio = np.pad(audio, (0, int(samples) - len(audio)))
    else:
        audio = audio[:int(samples)]
    return torch.from_numpy(audio).float().view(1, 1, -1)


def stft_mag(x, n_fft):
    hop = n_fft // 4
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=window, return_complex=True,
    )
    return spec.abs()


def neural_waveform_loss(target, pred):
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
    loss = spectral + 0.18 * convergence + 0.16 * flux + 0.08 * envelope + 0.012 * weak_wave
    return loss, {
        "spectral": float(spectral.detach()),
        "spectral_convergence": float(convergence.detach()),
        "flux": float(flux.detach()),
        "envelope": float(envelope.detach()),
        "weak_wave": float(weak_wave.detach()),
    }


def semitones(a, b):
    if a <= 0 or b <= 0:
        return 0.0
    return abs(12.0 * math.log2(float(b) / float(a)))


def bucket_name(distance):
    for name, lo, hi in PITCH_BUCKETS:
        if lo <= float(distance) < hi:
            return name
    return "extreme"


def build_manifest_index(manifest_path, voicebank_root):
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entries = payload.get("entries") or []
    usable = []
    for item in entries:
        if item.get("status") == "error" or not item.get("cache") or not item.get("relative_wav"):
            continue
        cache = (Path(voicebank_root) / str(item["cache"])).resolve()
        wav = (Path(voicebank_root) / str(item["relative_wav"])).resolve()
        if not cache.is_file() or not wav.is_file():
            continue
        copy = dict(item)
        copy["_cache"] = cache
        copy["_alias"] = str(item.get("base_alias") or item.get("alias") or "")
        copy["_f0"] = float(item.get("median_f0_hz", 0.0) or 0.0)
        usable.append(copy)
    return usable


def alias_split(entries, validation_fraction=0.12):
    aliases = sorted({x["_alias"] for x in entries if x["_alias"]})
    rng = random.Random(20260903)
    rng.shuffle(aliases)
    val_n = max(1, int(round(len(aliases) * float(validation_fraction)))) if len(aliases) > 1 else 0
    val_aliases = set(aliases[:val_n])
    train = [x for x in entries if x["_alias"] not in val_aliases]
    val = [x for x in entries if x["_alias"] in val_aliases]
    return train, val, val_aliases


def build_pairs(entries):
    groups = {}
    for item in entries:
        if item["_alias"] and item["_f0"] > 0:
            groups.setdefault(item["_alias"], []).append(item)
    buckets = {name: [] for name, _, _ in PITCH_BUCKETS}
    for alias, items in groups.items():
        items = sorted(items, key=lambda x: x["_f0"])
        for i, src in enumerate(items):
            for j, tgt in enumerate(items):
                if i == j:
                    continue
                if int(src.get("subbank_index", -1)) == int(tgt.get("subbank_index", -1)):
                    continue
                distance = semitones(src["_f0"], tgt["_f0"])
                if distance < 0.35:
                    continue
                buckets[bucket_name(distance)].append({
                    "alias": alias,
                    "source": src,
                    "target": tgt,
                    "semitones": distance,
                })
    rng = random.Random(20260903 + 1)
    for values in buckets.values():
        rng.shuffle(values)
    ordered = []
    cursor = 0
    while True:
        added = False
        for name, _, _ in PITCH_BUCKETS:
            values = buckets[name]
            if cursor < len(values):
                ordered.append(values[cursor])
                added = True
        if not added:
            break
        cursor += 1
    return ordered, {k: len(v) for k, v in buckets.items()}


def prepare_condition(engine, source, target_f0, seed):
    frames = int(target_f0.shape[-1])
    latent = F.interpolate(source["latent"], size=frames, mode="linear", align_corners=False)
    detail = F.interpolate(source["detail"], size=frames, mode="linear", align_corners=False)
    torch.manual_seed(int(seed))
    with torch.no_grad():
        structure, aux = engine.decoder(
            target_f0, latent, detail=detail, synthesis_sample_rate=SAMPLE_RATE, return_aux=True,
        )
    conditioning = build_neural_conditioning(latent, detail, target_f0, aux)
    return conditioning, structure.detach()


def make_model(engine, item):
    sample = load_cache(item["_cache"], engine.device)
    conditioning, _ = prepare_condition(engine, sample, sample["f0"], stable_seed(str(item["_cache"]), "shape"))
    model = YuazNeuralWaveformDecoder(condition_channels=int(conditioning.shape[1]))
    if model.output_hop != int(round(SAMPLE_RATE * engine.hop / engine.sr)):
        raise RuntimeError(
            f"neural decoder output hop {model.output_hop} does not match Yuaz frame hop "
            f"{SAMPLE_RATE * engine.hop / engine.sr:.1f}"
        )
    return model.to(engine.device)


def train_native_epoch(engine, model, optimizer, items, limit, epoch):
    model.train()
    rng = random.Random(20260903 + epoch)
    items = list(items)
    rng.shuffle(items)
    items = items[: min(int(limit), len(items))]
    total = 0.0
    count = 0
    for idx, item in enumerate(items, 1):
        source = load_cache(item["_cache"], engine.device)
        conditioning, structure = prepare_condition(
            engine, source, source["f0"], stable_seed(str(item["_cache"]), "native", epoch)
        )
        pred = model(conditioning, structure)
        target = read_fullband_target(item.get("voicebank_root") or "", item, pred.shape[-1])
        # manifest entries do not always store voicebank_root; caller patches it below.
        target = target.to(engine.device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = neural_waveform_loss(target, pred)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        total += float(loss.detach())
        count += 1
        if idx == 1 or idx % 16 == 0 or idx == len(items):
            print(f"native [{idx}/{len(items)}] loss={float(loss.detach()):.4f} spectral={parts['spectral']:.4f}", flush=True)
    return total / max(1, count)


def train_pair_epoch(engine, model, optimizer, pairs, limit, epoch):
    model.train()
    rng = random.Random(20260903 + 100 + epoch)
    pairs = list(pairs)
    rng.shuffle(pairs)
    pairs = pairs[: min(int(limit), len(pairs))]
    total = 0.0
    count = 0
    bucket_counts = {}
    for idx, pair in enumerate(pairs, 1):
        src = load_cache(pair["source"]["_cache"], engine.device)
        tgt = load_cache(pair["target"]["_cache"], engine.device)
        conditioning, structure = prepare_condition(
            engine, src, tgt["f0"], stable_seed(pair["alias"], pair["semitones"], epoch)
        )
        pred = model(conditioning, structure)
        target = read_fullband_target(pair["target"]["voicebank_root"], pair["target"], pred.shape[-1]).to(engine.device)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = neural_waveform_loss(target, pred)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        total += float(loss.detach())
        count += 1
        name = bucket_name(pair["semitones"])
        bucket_counts[name] = bucket_counts.get(name, 0) + 1
        if idx == 1 or idx % 16 == 0 or idx == len(pairs):
            print(
                f"pair [{idx}/{len(pairs)}] {name} {pair['semitones']:.1f}st "
                f"alias={pair['alias']} loss={float(loss.detach()):.4f}", flush=True,
            )
    return total / max(1, count), bucket_counts


def validate(engine, model, items, limit=48):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for item in items[: min(int(limit), len(items))]:
            sample = load_cache(item["_cache"], engine.device)
            conditioning, structure = prepare_condition(
                engine, sample, sample["f0"], stable_seed(str(item["_cache"]), "validation")
            )
            pred = model(conditioning, structure)
            target = read_fullband_target(item["voicebank_root"], item, pred.shape[-1]).to(engine.device)
            loss, _ = neural_waveform_loss(target, pred)
            if torch.isfinite(loss):
                total += float(loss)
                count += 1
    return total / max(1, count) if count else None


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
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    voicebank = Path(args.voicebank).expanduser().resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = Path(args.manifest).expanduser().resolve() if args.manifest else voicebank / ".yuaz-0.2.8ai14" / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"manifest not found: {manifest}")

    engine = YuazDDSPResamplerEngine(
        config["yuaz_repo"], config["checkpoint"], output_sr=SAMPLE_RATE,
        registry_path=config.get("registry_path"), ddsp_synthesis_sr=SAMPLE_RATE,
    )
    entries = build_manifest_index(manifest, voicebank)
    if not entries:
        raise RuntimeError("no usable cached voicebank entries")
    for item in entries:
        item["voicebank_root"] = str(voicebank)

    train_items, val_items, val_aliases = alias_split(entries)
    train_pairs, pair_buckets = build_pairs(train_items)
    val_pairs, val_pair_buckets = build_pairs(val_items)
    print(f"native train={len(train_items)} val={len(val_items)} held-out aliases={len(val_aliases)}")
    print(f"train pair buckets={pair_buckets}")
    print(f"validation pair buckets={val_pair_buckets}")

    model = make_model(engine, train_items[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-5)
    history = []

    for epoch in range(int(args.native_epochs)):
        train_loss = train_native_epoch(engine, model, optimizer, train_items, args.native_limit, epoch)
        val_loss = validate(engine, model, val_items)
        history.append({"stage": "native", "epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss})
        print(f"native epoch {epoch + 1}: train={train_loss:.4f} val={val_loss}")

    for epoch in range(int(args.pair_epochs)):
        train_loss, seen = train_pair_epoch(engine, model, optimizer, train_pairs, args.pair_limit, epoch)
        val_loss = validate(engine, model, val_items)
        history.append({"stage": "multipitch", "epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss, "seen_buckets": seen})
        print(f"multipitch epoch {epoch + 1}: train={train_loss:.4f} val={val_loss} buckets={seen}")

    output = Path(args.output).expanduser().resolve() if args.output else root / "control_models" / "neural-waveform-v0.3.0.pt"
    save_neural_waveform_decoder(output, model, {
        "version": "0.3.0",
        "sample_rate": SAMPLE_RATE,
        "voicebank": str(voicebank),
        "manifest": str(manifest),
        "validation_aliases": sorted(val_aliases),
        "train_pair_buckets": pair_buckets,
        "validation_pair_buckets": val_pair_buckets,
        "history": history,
        "training_definition": "native-pitch reconstruction + alias-isolated multipitch cross-recording",
        "loss": "MR-STFT + spectral convergence + spectral flux + envelope + weak waveform auxiliary",
    })
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
