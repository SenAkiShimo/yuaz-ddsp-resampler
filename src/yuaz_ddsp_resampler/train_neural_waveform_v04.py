#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path

import librosa
import numpy as np
import torch

from . import train_neural_waveform_v3 as v3
from . import train_neural_waveform_v4 as v4
from .neural_waveform import save_neural_waveform_decoder
from .train_neural_waveform import (
    PITCH_BUCKETS,
    SAMPLE_RATE,
    alias_split,
    bucket_name,
    build_manifest_index,
    build_pairs,
    load_cache,
    read_fullband_target,
    resolve_manifest,
)

VERSION = "0.4.0"
TRAINING_GENERATION = "v0.4-robust-pairs"
SYNTHETIC_SHIFTS = {
    "near": (-2.0, 2.0),
    "medium": (-5.0, 5.0),
    "far": (-9.0, 9.0),
    "extreme": (-14.0, 14.0),
}

def voicebank_id(path):
    path = Path(path).expanduser().resolve()
    slug = re.sub(r"[^0-9A-Za-z._-]+", "-", path.name).strip("-") or "voicebank"
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}"

def pair_bucket_counts(pairs):
    out = {name: 0 for name, _, _ in PITCH_BUCKETS}
    for pair in pairs:
        out[bucket_name(pair["semitones"])] += 1
    return out

def pair_mode_counts(pairs):
    out = {"recorded": 0, "synthetic": 0}
    for pair in pairs:
        out["synthetic" if pair.get("synthetic") else "recorded"] += 1
    return out

def _synthetic_pair(source, shift):
    return {
        "alias": source["_alias"],
        "source": source,
        "target": source,
        "semitones": abs(float(shift)),
        "signed_semitones": float(shift),
        "synthetic": True,
    }

def top_up_pairs(entries, recorded_pairs, minimum_per_bucket, seed):
    pairs = list(recorded_pairs)
    counts = pair_bucket_counts(pairs)
    candidates = [x for x in entries if float(x.get("_f0", 0.0) or 0.0) > 1.0]
    if not candidates:
        raise RuntimeError("voicebank has no voiced entries available for synthetic cross-pitch training")
    rng = random.Random(int(seed))
    candidates = list(candidates)
    rng.shuffle(candidates)
    cursor = 0
    for name, _, _ in PITCH_BUCKETS:
        need = max(0, int(minimum_per_bucket) - int(counts.get(name, 0)))
        shifts = SYNTHETIC_SHIFTS[name]
        for idx in range(need):
            source = candidates[cursor % len(candidates)]
            cursor += 1
            pairs.append(_synthetic_pair(source, shifts[idx % len(shifts)]))
    return pairs

def target_f0_for_pair(pair, source_cache, device):
    if pair.get("synthetic"):
        factor = 2.0 ** (float(pair["signed_semitones"]) / 12.0)
        return source_cache["f0"] * float(factor)
    target = load_cache(pair["target"]["_cache"], device)
    return target["f0"]

def synthetic_target_audio(pair, samples, device):
    source = pair["source"]
    target = read_fullband_target(source["voicebank_root"], source, int(samples))
    y = target.detach().cpu().numpy().reshape(-1).astype(np.float32)
    shift = float(pair["signed_semitones"])
    try:
        shifted = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=shift, bins_per_octave=12, res_type="soxr_hq")
    except Exception:
        shifted = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=shift, bins_per_octave=12)
    shifted = np.asarray(shifted, dtype=np.float32)
    if shifted.size < int(samples):
        shifted = np.pad(shifted, (0, int(samples) - shifted.size))
    else:
        shifted = shifted[: int(samples)]
    return torch.from_numpy(shifted).to(device).view(1, 1, -1)

def pair_target_audio(pair, samples, device):
    if pair.get("synthetic"):
        return synthetic_target_audio(pair, samples, device)
    target = pair["target"]
    return read_fullband_target(target["voicebank_root"], target, int(samples)).to(device)

def train_pair_epoch(engine, model, optimizer, pairs, native_items, limit, epoch, prepare_fn, loss_fn, stage_tag):
    model.train()
    rng = random.Random(20260915 + 100 + int(epoch))
    selected = list(pairs)
    native_items = list(native_items)
    rng.shuffle(selected)
    rng.shuffle(native_items)
    selected = selected[: min(int(limit), len(selected))]
    if not selected:
        raise RuntimeError("cross-pitch training set is empty")
    total = pair_total = native_total = 0.0
    count = 0
    bucket_counts = {}
    mode_counts = {"recorded": 0, "synthetic": 0}
    for idx, pair in enumerate(selected, 1):
        source = load_cache(pair["source"]["_cache"], engine.device)
        target_f0 = target_f0_for_pair(pair, source, engine.device)
        conditioning, structure = prepare_fn(engine, source, pair["source"], target_f0, v3.stable_seed(pair["alias"], pair["semitones"], stage_tag, epoch))
        pred = model(conditioning, structure)
        target = pair_target_audio(pair, pred.shape[-1], engine.device)
        pair_loss, _ = loss_fn(target, pred)
        anchor_item = native_items[(idx - 1) % len(native_items)]
        anchor = load_cache(anchor_item["_cache"], engine.device)
        anchor_condition, anchor_structure = prepare_fn(engine, anchor, anchor_item, anchor["f0"], v3.stable_seed(str(anchor_item["_cache"]), stage_tag, "rehearsal", epoch, idx))
        anchor_pred = model(anchor_condition, anchor_structure)
        anchor_target = read_fullband_target(anchor_item["voicebank_root"], anchor_item, anchor_pred.shape[-1]).to(engine.device)
        native_loss, _ = loss_fn(anchor_target, anchor_pred)
        loss = v3.PAIR_WEIGHT * pair_loss + v3.NATIVE_REHEARSAL_WEIGHT * native_loss
        if not torch.isfinite(loss):
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        total += float(loss.detach()); pair_total += float(pair_loss.detach()); native_total += float(native_loss.detach()); count += 1
        bucket = bucket_name(pair["semitones"])
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        mode = "synthetic" if pair.get("synthetic") else "recorded"
        mode_counts[mode] += 1
        if idx == 1 or idx % 16 == 0 or idx == len(selected):
            print(f"{stage_tag} [{idx}/{len(selected)}] {bucket} {pair['semitones']:.1f}st mode={mode} mix={float(loss.detach()):.4f} pair={float(pair_loss.detach()):.4f} native={float(native_loss.detach()):.4f}", flush=True)
    if count == 0:
        raise RuntimeError("all cross-pitch training batches were rejected")
    denom = float(count)
    return total / denom, pair_total / denom, native_total / denom, bucket_counts, mode_counts

def validate_pairs(engine, model, pairs, limit, prepare_fn, loss_fn, stage_tag):
    model.eval()
    selected = list(pairs)
    random.Random(20260915 + 900).shuffle(selected)
    selected = selected[: min(int(limit), len(selected))]
    if not selected:
        raise RuntimeError("cross-pitch validation set is empty")
    totals = {name: 0.0 for name, _, _ in PITCH_BUCKETS}; counts = {name: 0 for name, _, _ in PITCH_BUCKETS}; modes = {"recorded": 0, "synthetic": 0}
    total = 0.0; count = 0
    with torch.no_grad():
        for pair in selected:
            source = load_cache(pair["source"]["_cache"], engine.device)
            target_f0 = target_f0_for_pair(pair, source, engine.device)
            conditioning, structure = prepare_fn(engine, source, pair["source"], target_f0, v3.stable_seed(pair["alias"], pair["semitones"], stage_tag, "validation"))
            pred = model(conditioning, structure)
            target = pair_target_audio(pair, pred.shape[-1], engine.device)
            loss, _ = loss_fn(target, pred)
            if not torch.isfinite(loss):
                continue
            value = float(loss); bucket = bucket_name(pair["semitones"])
            total += value; count += 1; totals[bucket] += value; counts[bucket] += 1
            modes["synthetic" if pair.get("synthetic") else "recorded"] += 1
    if count == 0:
        raise RuntimeError("all cross-pitch validation batches were rejected")
    buckets = {name: {"loss": totals[name] / counts[name] if counts[name] else None, "count": counts[name]} for name, _, _ in PITCH_BUCKETS}
    return {"overall": total / count, "count": count, "buckets": buckets, "modes": modes}

def audit_payload(voicebank, entries, train_items, val_items, recorded_train, recorded_val, robust_train, robust_val):
    f0s = np.asarray([float(x.get("_f0", 0.0) or 0.0) for x in entries], dtype=np.float64); f0s = f0s[f0s > 1.0]
    subbanks = sorted({int(x.get("subbank_index", -1)) for x in entries}); aliases = sorted({x.get("_alias") for x in entries if x.get("_alias")})
    return {"version": VERSION, "training_generation": TRAINING_GENERATION, "voicebank": str(Path(voicebank).resolve()), "voicebank_id": voicebank_id(voicebank), "entries": len(entries), "aliases": len(aliases), "subbanks": len(subbanks), "train_entries": len(train_items), "validation_entries": len(val_items), "median_f0_hz": float(np.median(f0s)) if f0s.size else None, "f0_min_hz": float(np.min(f0s)) if f0s.size else None, "f0_max_hz": float(np.max(f0s)) if f0s.size else None, "recorded_train_pairs": pair_bucket_counts(recorded_train), "recorded_validation_pairs": pair_bucket_counts(recorded_val), "robust_train_pairs": pair_bucket_counts(robust_train), "robust_validation_pairs": pair_bucket_counts(robust_val), "train_pair_modes": pair_mode_counts(robust_train), "validation_pair_modes": pair_mode_counts(robust_val)}

def checkpoint_paths(stage_dir, prefix):
    return {"final": stage_dir / f"{prefix}.pt", "native_best": stage_dir / f"{prefix}-native-best.pt", "multipitch_best": stage_dir / f"{prefix}-multipitch-best.pt", "pareto_best": stage_dir / f"{prefix}-pareto-best.pt"}

def save_stage_checkpoint(path, model, metadata_builder, voicebank, manifest, val_aliases, pair_buckets, val_pair_buckets, history, role, bank_id, pair_modes, validation_pair_modes, best_native, best_pair, pareto_native_limit):
    metadata = metadata_builder(voicebank, manifest, val_aliases, pair_buckets, val_pair_buckets, history, role, best_native=best_native, best_pair=best_pair, pareto_native_limit=pareto_native_limit)
    metadata.update({"version": VERSION, "training_generation": TRAINING_GENERATION, "voicebank_id": bank_id, "pair_modes": dict(pair_modes), "validation_pair_modes": dict(validation_pair_modes), "synthetic_pair_target": "librosa.effects.pitch_shift", "synthetic_pair_shifts": SYNTHETIC_SHIFTS})
    save_neural_waveform_decoder(path, model, metadata)

def validate_checkpoint_voicebank(path, voicebank, bank_id):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True); metadata = dict(payload.get("metadata") or {})
    recorded_bank = Path(str(metadata.get("voicebank") or "")).expanduser().resolve(); requested_bank = Path(voicebank).expanduser().resolve()
    if recorded_bank != requested_bank:
        raise RuntimeError(f"warm-start voicebank mismatch: checkpoint={recorded_bank} requested={requested_bank}")
    if str(metadata.get("voicebank_id") or "") != str(bank_id):
        raise RuntimeError("warm-start voicebank id mismatch")
    return metadata

def run_stage(*, engine, model, optimizer, train_items, val_items, robust_train, robust_val, args, stage_name, prepare_fn, loss_fn, metadata_builder, paths, bank_id, voicebank, manifest, val_aliases, train_pair_buckets, val_pair_buckets, train_modes, val_modes):
    history = []; best_native = best_pair = best_pareto = pareto_native_limit = None
    for epoch in range(int(args.native_epochs)):
        train_loss = v3.train_native_epoch_v3(engine, model, optimizer, train_items, args.native_limit, epoch)
        native_val = v3.validate_native_v3(engine, model, val_items, args.native_val_limit)
        history.append({"stage": f"native-{stage_name}", "epoch": epoch + 1, "train_loss": train_loss, "native_validation_loss": native_val})
        print(f"native-{stage_name} epoch {epoch + 1}: train={train_loss:.4f} native_val={native_val}")
        if native_val is not None and (best_native is None or native_val < best_native):
            best_native = native_val; pareto_native_limit = best_native * (1.0 + v3.PARETO_NATIVE_DEGRADATION)
            save_stage_checkpoint(paths["native_best"], model, metadata_builder, voicebank, manifest, val_aliases, train_pair_buckets, val_pair_buckets, history, f"native-best-{stage_name}", bank_id, train_modes, val_modes, best_native, best_pair, pareto_native_limit)
    if best_native is None:
        raise RuntimeError(f"{stage_name} native stage did not produce a valid checkpoint")
    for group in optimizer.param_groups:
        group["lr"] = float(args.pair_lr)
    print(f"pair-{stage_name} stage lr={float(args.pair_lr):.6g} native_limit={pareto_native_limit:.6f} modes={train_modes} buckets={train_pair_buckets}", flush=True)
    for epoch in range(int(args.pair_epochs)):
        mix_loss, pair_train, native_rehearsal, seen, seen_modes = train_pair_epoch(engine, model, optimizer, robust_train, train_items, args.pair_limit, epoch, prepare_fn, loss_fn, f"pair-{stage_name}")
        native_val = v3.validate_native_v3(engine, model, val_items, args.native_val_limit)
        pair_val = validate_pairs(engine, model, robust_val, args.pair_val_limit, prepare_fn, loss_fn, f"pair-{stage_name}")
        pair_overall = pair_val["overall"]
        eligible = pair_overall is not None and native_val is not None and pareto_native_limit is not None and native_val <= pareto_native_limit
        history.append({"stage": f"multipitch-{stage_name}", "epoch": epoch + 1, "train_loss": mix_loss, "pair_train_loss": pair_train, "native_rehearsal_loss": native_rehearsal, "native_validation_loss": native_val, "cross_pitch_validation": pair_val, "pareto_eligible": bool(eligible), "seen_buckets": seen, "seen_modes": seen_modes})
        print(f"multipitch-{stage_name} epoch {epoch + 1}: mix={mix_loss:.4f} pair={pair_train:.4f} rehearsal={native_rehearsal:.4f} native_val={native_val} cross_val={pair_overall} pareto={eligible} buckets={pair_val['buckets']} modes={pair_val['modes']}", flush=True)
        if pair_overall is not None and (best_pair is None or pair_overall < best_pair):
            best_pair = pair_overall
            save_stage_checkpoint(paths["multipitch_best"], model, metadata_builder, voicebank, manifest, val_aliases, train_pair_buckets, val_pair_buckets, history, f"multipitch-best-{stage_name}", bank_id, train_modes, val_modes, best_native, best_pair, pareto_native_limit)
        if eligible and (best_pareto is None or pair_overall < best_pareto):
            best_pareto = pair_overall
            save_stage_checkpoint(paths["pareto_best"], model, metadata_builder, voicebank, manifest, val_aliases, train_pair_buckets, val_pair_buckets, history, f"pareto-best-{stage_name}", bank_id, train_modes, val_modes, best_native, best_pair, pareto_native_limit)
    save_stage_checkpoint(paths["final"], model, metadata_builder, voicebank, manifest, val_aliases, train_pair_buckets, val_pair_buckets, history, f"final-{stage_name}", bank_id, train_modes, val_modes, best_native, best_pair, pareto_native_limit)
    if best_pareto is None:
        raise RuntimeError(f"{stage_name} produced no Pareto-valid checkpoint; refusing to continue")
    print(f"recommended {stage_name} checkpoint: {paths['pareto_best']}")
    return paths["pareto_best"], best_pareto

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2])); parser.add_argument("--voicebank", required=True); parser.add_argument("--manifest", default=None); parser.add_argument("--output-dir", default=None)
    parser.add_argument("--native-epochs", type=int, default=3); parser.add_argument("--pair-epochs", type=int, default=3); parser.add_argument("--native-limit", type=int, default=512); parser.add_argument("--pair-limit", type=int, default=512); parser.add_argument("--native-val-limit", type=int, default=48); parser.add_argument("--pair-val-limit", type=int, default=96); parser.add_argument("--min-train-per-bucket", type=int, default=128); parser.add_argument("--min-val-per-bucket", type=int, default=24); parser.add_argument("--lr", type=float, default=2e-4); parser.add_argument("--pair-lr", type=float, default=v3.PAIR_LR); parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve(); voicebank = Path(args.voicebank).expanduser().resolve(); manifest = resolve_manifest(voicebank, args.manifest); config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    engine = v3.YuazDDSPResamplerEngine(config["yuaz_repo"], config["checkpoint"], output_sr=SAMPLE_RATE, registry_path=config.get("registry_path"), ddsp_synthesis_sr=SAMPLE_RATE)
    entries = build_manifest_index(manifest, voicebank)
    if not entries: raise RuntimeError("no usable cached voicebank entries")
    train_items, val_items, val_aliases = alias_split(entries)
    if not train_items or not val_items: raise RuntimeError("train/validation split is empty")
    recorded_train, _ = build_pairs(train_items); recorded_val, _ = build_pairs(val_items)
    robust_train = top_up_pairs(train_items, recorded_train, args.min_train_per_bucket, 20260915); robust_val = top_up_pairs(val_items, recorded_val, args.min_val_per_bucket, 20260916)
    audit = audit_payload(voicebank, entries, train_items, val_items, recorded_train, recorded_val, robust_train, robust_val); print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    bank_id = audit["voicebank_id"]; out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "control_models" / "v0.4" / bank_id; out_dir.mkdir(parents=True, exist_ok=True); (out_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.audit_only: print(f"audit complete: {out_dir / 'audit.json'}"); return
    train_pair_buckets = pair_bucket_counts(robust_train); val_pair_buckets = pair_bucket_counts(robust_val); train_modes = pair_mode_counts(robust_train); val_modes = pair_mode_counts(robust_val)
    os.environ.pop(v4.V4_WARM_START_ENV, None)
    v3_dir = out_dir / "v3"; v3_dir.mkdir(parents=True, exist_ok=True); v3_paths = checkpoint_paths(v3_dir, "neural-waveform-v0.4-v3")
    v3_model = v3.make_model_v3(engine, train_items[0]); v3_optimizer = torch.optim.AdamW(v3_model.parameters(), lr=float(args.lr), weight_decay=1e-5)
    v3_pareto, _ = run_stage(engine=engine, model=v3_model, optimizer=v3_optimizer, train_items=train_items, val_items=val_items, robust_train=robust_train, robust_val=robust_val, args=args, stage_name="v3-robust", prepare_fn=v3.prepare_condition_v3, loss_fn=v3.neural_waveform_loss_v3, metadata_builder=v3.metadata, paths=v3_paths, bank_id=bank_id, voicebank=voicebank, manifest=manifest, val_aliases=val_aliases, train_pair_buckets=train_pair_buckets, val_pair_buckets=val_pair_buckets, train_modes=train_modes, val_modes=val_modes)
    validate_checkpoint_voicebank(v3_pareto, voicebank, bank_id); print(f"v4 warm-start source verified: {v3_pareto}", flush=True)
    os.environ[v4.V4_WARM_START_ENV] = str(v3_pareto); v4.install_v4_overrides()
    v4_dir = out_dir / "v4"; v4_dir.mkdir(parents=True, exist_ok=True); v4_paths = checkpoint_paths(v4_dir, "neural-waveform-v0.4-v4")
    v4_model = v4.make_model_v4(engine, train_items[0]); v4_optimizer = torch.optim.AdamW(v4_model.parameters(), lr=float(args.lr), weight_decay=1e-5)
    v4_pareto, _ = run_stage(engine=engine, model=v4_model, optimizer=v4_optimizer, train_items=train_items, val_items=val_items, robust_train=robust_train, robust_val=robust_val, args=args, stage_name="v4-robust", prepare_fn=v4.prepare_condition_v4, loss_fn=v4.neural_waveform_loss_v4, metadata_builder=v4.metadata_v4, paths=v4_paths, bank_id=bank_id, voicebank=voicebank, manifest=manifest, val_aliases=val_aliases, train_pair_buckets=train_pair_buckets, val_pair_buckets=val_pair_buckets, train_modes=train_modes, val_modes=val_modes)
    validate_checkpoint_voicebank(v4_pareto, voicebank, bank_id)
    recommended = out_dir / "recommended.json"; recommended.write_text(json.dumps({"version": VERSION, "voicebank": str(voicebank), "voicebank_id": bank_id, "checkpoint": str(v4_pareto), "v3_warm_start": str(v3_pareto), "pair_buckets": train_pair_buckets, "pair_modes": train_modes}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"v0.4 robust training complete for {bank_id}"); print(f"recommended checkpoint: {v4_pareto}"); print(f"selection record: {recommended}")

if __name__ == "__main__":
    main()
