#!/usr/bin/env python3
"""Blind listening export for v0.4 robust-v3, robust-v4 and HiFi checkpoints."""

import argparse
import csv
import json
import random
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from .core import YuazDDSPResamplerEngine, stable_seed
from .neural_waveform import load_neural_waveform_decoder
from .train_neural_waveform import (
    SAMPLE_RATE,
    alias_split,
    build_manifest_index,
    load_cache,
    read_fullband_target,
    resolve_manifest,
)
from .train_neural_waveform_v3 import prepare_condition_v3
from .train_neural_waveform_v4 import prepare_condition_v4
from .train_neural_waveform_hifi import prepare_condition_hifi
from .train_neural_waveform_v04 import voicebank_id


GENERATION_PREPARE = {
    "conditioned-v3": prepare_condition_v3,
    "conditioned-v4": prepare_condition_v4,
    "conditioned-v4-hifi": prepare_condition_hifi,
}

TRACKS = (
    ("v3_robust", "v3", "neural-waveform-v0.4-v3-pareto-best.pt", "conditioned-v3"),
    ("v4_robust", "v4", "neural-waveform-v0.4-v4-pareto-best.pt", "conditioned-v4"),
    ("hifi", "hifi", "neural-waveform-v0.4-hifi-pareto-best.pt", "conditioned-v4-hifi"),
)


def safe_name(text):
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("_")
    return text[:72] or "sample"


def parse_shifts(text):
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise argparse.ArgumentTypeError("at least one semitone shift is required")
    return values


def write_audio(path, tensor_or_array):
    if hasattr(tensor_or_array, "detach"):
        x = tensor_or_array.detach().cpu().numpy()
    else:
        x = np.asarray(tensor_or_array)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = np.nan_to_num(x)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(x, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_24")


def resolve_track_checkpoints(root, voicebank):
    bank_id = voicebank_id(voicebank)
    bank_root = root / "control_models" / "v0.4" / bank_id
    resolved = {}
    missing = []
    for label, stage_dir, filename, expected_generation in TRACKS:
        path = bank_root / stage_dir / filename
        if not path.is_file():
            missing.append(str(path))
            continue
        model, metadata = load_neural_waveform_decoder(path, device="cpu")
        del model
        trained_bank = Path(str(metadata.get("voicebank") or "")).expanduser().resolve()
        if trained_bank != voicebank:
            raise RuntimeError(
                f"{label} checkpoint voicebank mismatch: checkpoint={trained_bank} requested={voicebank}"
            )
        generation = str(metadata.get("trainer_generation") or "")
        if generation != expected_generation:
            raise RuntimeError(
                f"{label} generation mismatch: expected {expected_generation}, got {generation or 'missing'}"
            )
        resolved[label] = {
            "path": path,
            "generation": generation,
            "metadata": metadata,
        }
    if missing:
        raise RuntimeError(
            "required v0.4 checkpoints are missing:\n  " + "\n  ".join(missing)
            + "\nRun robust v0.4 training and HiFi fine-tuning for this voicebank first."
        )
    return bank_id, resolved


def load_tracks(engine, resolved):
    tracks = {}
    for label, info in resolved.items():
        model, metadata = load_neural_waveform_decoder(info["path"], device=engine.device)
        generation = str(metadata.get("trainer_generation") or "")
        prepare_fn = GENERATION_PREPARE.get(generation)
        if prepare_fn is None:
            raise RuntimeError(f"unsupported listening-test generation: {generation}")
        tracks[label] = {
            "model": model,
            "metadata": metadata,
            "prepare_fn": prepare_fn,
            "path": info["path"],
        }
    return tracks


def target_f0(source_cache, semitones):
    factor = 2.0 ** (float(semitones) / 12.0)
    return source_cache["f0"] * float(factor)


def pitchshift_reference(item, samples, semitones, device):
    original = read_fullband_target(item["voicebank_root"], item, samples)
    y = original.detach().cpu().numpy().reshape(-1).astype(np.float32)
    if abs(float(semitones)) < 1e-9:
        return original.to(device)
    try:
        shifted = librosa.effects.pitch_shift(
            y, sr=SAMPLE_RATE, n_steps=float(semitones), bins_per_octave=12, res_type="soxr_hq"
        )
    except Exception:
        shifted = librosa.effects.pitch_shift(
            y, sr=SAMPLE_RATE, n_steps=float(semitones), bins_per_octave=12
        )
    shifted = np.asarray(shifted, dtype=np.float32)
    if shifted.size < samples:
        shifted = np.pad(shifted, (0, samples - shifted.size))
    else:
        shifted = shifted[:samples]
    return torch.from_numpy(shifted).to(device).view(1, 1, -1)


def render_track(engine, track, source_cache, item, target_f0_tensor, seed):
    prepare_fn = track["prepare_fn"]
    conditioning, structure, raw_structure = prepare_fn(
        engine,
        source_cache,
        item,
        target_f0_tensor,
        seed,
        return_raw=True,
    )
    model = track["model"]
    model.eval()
    with torch.inference_mode():
        pred = model(conditioning, structure)
    return pred, raw_structure, structure


def deterministic_blind_order(alias, semitones, labels):
    labels = list(labels)
    rng = random.Random(stable_seed(str(alias), float(semitones), "v04-blind-listening-order"))
    rng.shuffle(labels)
    return labels


def write_scorecard(path, rows):
    fieldnames = [
        "sample_id",
        "alias",
        "shift_st",
        "A_clarity_1_5",
        "A_highband_1_5",
        "B_clarity_1_5",
        "B_highband_1_5",
        "C_clarity_1_5",
        "C_highband_1_5",
        "best_letter",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sample_id": row["sample_id"],
                "alias": row["alias"],
                "shift_st": row["shift_st"],
            })


def main():
    parser = argparse.ArgumentParser(
        description="Export blind, matched v0.4 V3/V4/HiFi listening comparisons."
    )
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--voicebank", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--shifts", type=parse_shifts, default=parse_shifts("0,5,9,14"))
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    voicebank = Path(args.voicebank).expanduser().resolve()
    if not voicebank.is_dir():
        raise RuntimeError(f"voicebank folder not found: {voicebank}")
    manifest = resolve_manifest(voicebank, args.manifest)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))

    bank_id, resolved = resolve_track_checkpoints(root, voicebank)
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root / "listening_tests" / f"{bank_id}-v04"
    )
    labeled_dir = out_dir / "labeled"
    blind_dir = out_dir / "blind"
    reference_dir = out_dir / "reference"
    for path in (labeled_dir, blind_dir, reference_dir):
        path.mkdir(parents=True, exist_ok=True)

    engine = YuazDDSPResamplerEngine(
        config["yuaz_repo"],
        config["checkpoint"],
        output_sr=SAMPLE_RATE,
        registry_path=config.get("registry_path"),
        ddsp_synthesis_sr=SAMPLE_RATE,
    )
    tracks = load_tracks(engine, resolved)

    entries = build_manifest_index(manifest, voicebank)
    _, val_items, val_aliases = alias_split(entries)
    voiced = [x for x in val_items if float(x.get("_f0", 0.0) or 0.0) > 1.0]
    if not voiced:
        raise RuntimeError("no voiced held-out entries available for listening export")
    selected_items = voiced[: max(1, int(args.samples))]

    report = {
        "version": "0.4.0",
        "purpose": "blind matched comparison of robust v3, robust v4, and HiFi",
        "voicebank": str(voicebank),
        "voicebank_id": bank_id,
        "manifest": str(manifest),
        "held_out_aliases": len(val_aliases),
        "shifts_semitones": [float(x) for x in args.shifts],
        "tracks": {
            label: {
                "checkpoint": str(track["path"]),
                "trainer_generation": track["metadata"].get("trainer_generation"),
                "checkpoint_role": track["metadata"].get("checkpoint_role"),
            }
            for label, track in tracks.items()
        },
        "samples": [],
    }
    blind_key = {}
    score_rows = []

    for item_index, item in enumerate(selected_items, 1):
        source_cache = load_cache(item["_cache"], engine.device)
        alias = str(item["_alias"])
        for shift in args.shifts:
            shift = float(shift)
            sample_id = f"{item_index:02d}__{safe_name(alias)}__{shift:+.1f}st"
            f0 = target_f0(source_cache, shift)
            seed = stable_seed(alias, shift, "v04-listening-render")

            rendered = {}
            structure_reference = None
            raw_reference = None
            for label, track in tracks.items():
                pred, raw_structure, structure = render_track(
                    engine, track, source_cache, item, f0, seed
                )
                rendered[label] = pred
                if label == "v4_robust":
                    raw_reference = raw_structure
                    structure_reference = structure
                write_audio(labeled_dir / f"{sample_id}__{label}.wav", pred)

            output_samples = int(next(iter(rendered.values())).shape[-1])
            original = read_fullband_target(
                item["voicebank_root"], item, output_samples
            ).to(engine.device)
            ref_shifted = pitchshift_reference(
                item, output_samples, shift, engine.device
            )
            write_audio(reference_dir / f"{sample_id}__source-original.wav", original)
            write_audio(reference_dir / f"{sample_id}__pitchshift-reference.wav", ref_shifted)
            if raw_reference is not None:
                write_audio(reference_dir / f"{sample_id}__ddsp-raw.wav", raw_reference)
            if structure_reference is not None:
                write_audio(reference_dir / f"{sample_id}__ddsp-structure.wav", structure_reference)

            blind_order = deterministic_blind_order(alias, shift, rendered.keys())
            letters = ("A", "B", "C")
            mapping = {}
            for letter, label in zip(letters, blind_order):
                mapping[letter] = label
                write_audio(blind_dir / f"{sample_id}__{letter}.wav", rendered[label])
            blind_key[sample_id] = mapping

            row = {
                "sample_id": sample_id,
                "alias": alias,
                "shift_st": shift,
                "source_wav": str(item["_wav"]),
                "blind_files": {
                    letter: str(blind_dir / f"{sample_id}__{letter}.wav")
                    for letter in letters
                },
                "reference_pitchshift_note": (
                    "Diagnostic only; librosa pitch shift is not a ground-truth target."
                ),
            }
            report["samples"].append(row)
            score_rows.append(row)
            print(f"exported {sample_id}: blind A/B/C + references", flush=True)

    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "blind_key.json").write_text(
        json.dumps(blind_key, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_scorecard(out_dir / "scorecard.csv", score_rows)

    instructions = """# v0.4 blind listening test

Listen to files in `blind/` first. For every sample, compare A/B/C without opening
`blind_key.json`.

Rate two things separately:

1. articulation/clarity: consonant edges, vowel identity, CV transition precision;
2. high-band fidelity: openness/air/detail without hiss, metallic texture, or source-pitch leakage.

Use `scorecard.csv` to record 1-5 scores and your preferred letter. The
`reference/` directory contains the original sample, a diagnostic pitch-shifted
reference, and the v4 DDSP raw/low-passed structure. The pitch-shift reference is
not ground truth.

After scoring all samples, open `blind_key.json` to reveal which letter was
v3 robust, v4 robust, or HiFi.
"""
    (out_dir / "README.md").write_text(instructions, encoding="utf-8")

    print(f"listening export complete: {out_dir}", flush=True)
    print("score blind/ first; reveal blind_key.json only after judging", flush=True)


if __name__ == "__main__":
    main()
