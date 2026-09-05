#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from .core import YuazDDSPResamplerEngine, stable_seed
from .neural_waveform import load_neural_waveform_decoder
from .train_neural_waveform import (
    SAMPLE_RATE,
    PITCH_BUCKETS,
    alias_split,
    bucket_name,
    build_manifest_index,
    build_pairs,
    load_cache,
    prepare_condition,
    read_fullband_target,
    resolve_manifest,
)
from .train_neural_waveform_v3 import prepare_condition_v3


def safe_name(text):
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("_")
    return text[:72] or "sample"


def write_audio(path, tensor_or_array):
    if hasattr(tensor_or_array, "detach"):
        x = tensor_or_array.detach().cpu().numpy()
    else:
        x = np.asarray(tensor_or_array)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = np.nan_to_num(x)
    x = np.clip(x, -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, SAMPLE_RATE, subtype="PCM_24")


def choose_pairs(val_pairs):
    chosen = {}
    for pair in val_pairs:
        name = bucket_name(pair["semitones"])
        if name not in chosen:
            chosen[name] = pair
    return chosen


def conditioning_function(metadata):
    return prepare_condition_v3 if metadata.get("trainer_generation") == "conditioned-v3" else prepare_condition


def render_pair(engine, model, pair, out_dir, prefix, prepare_fn):
    source = load_cache(pair["source"]["_cache"], engine.device)
    target_cache = load_cache(pair["target"]["_cache"], engine.device)
    if prepare_fn is prepare_condition_v3:
        conditioning, structure, raw_structure = prepare_fn(
            engine,
            source,
            pair["source"],
            target_cache["f0"],
            stable_seed(pair["alias"], pair["semitones"], "ab-export-v3"),
            return_raw=True,
        )
    else:
        conditioning, structure = prepare_fn(
            engine,
            source,
            pair["source"],
            target_cache["f0"],
            stable_seed(pair["alias"], pair["semitones"], "ab-export"),
        )
        raw_structure = structure
    with __import__("torch").inference_mode():
        neural = model(conditioning, structure)
    target = read_fullband_target(
        pair["target"]["voicebank_root"], pair["target"], neural.shape[-1]
    )
    source_original = read_fullband_target(
        pair["source"]["voicebank_root"], pair["source"], neural.shape[-1]
    )

    base = out_dir / prefix
    write_audio(base.with_name(base.name + "__source-original.wav"), source_original)
    write_audio(base.with_name(base.name + "__target-original.wav"), target)
    write_audio(base.with_name(base.name + "__ddsp-raw-48k.wav"), raw_structure)
    if prepare_fn is prepare_condition_v3:
        write_audio(base.with_name(base.name + "__ddsp-structure-48k.wav"), structure)
    write_audio(base.with_name(base.name + "__neural-48k.wav"), neural)
    return {
        "alias": pair["alias"],
        "bucket": bucket_name(pair["semitones"]),
        "semitones": float(pair["semitones"]),
        "source_wav": str(pair["source"]["_wav"]),
        "target_wav": str(pair["target"]["_wav"]),
        "prefix": prefix,
    }


def render_native(engine, model, item, out_dir, prefix, prepare_fn):
    sample = load_cache(item["_cache"], engine.device)
    if prepare_fn is prepare_condition_v3:
        conditioning, structure, raw_structure = prepare_fn(
            engine,
            sample,
            item,
            sample["f0"],
            stable_seed(str(item["_cache"]), "ab-native-v3"),
            return_raw=True,
        )
    else:
        conditioning, structure = prepare_fn(
            engine,
            sample,
            item,
            sample["f0"],
            stable_seed(str(item["_cache"]), "ab-native"),
        )
        raw_structure = structure
    with __import__("torch").inference_mode():
        neural = model(conditioning, structure)
    target = read_fullband_target(item["voicebank_root"], item, neural.shape[-1])
    base = out_dir / prefix
    write_audio(base.with_name(base.name + "__target-original.wav"), target)
    write_audio(base.with_name(base.name + "__ddsp-raw-48k.wav"), raw_structure)
    if prepare_fn is prepare_condition_v3:
        write_audio(base.with_name(base.name + "__ddsp-structure-48k.wav"), structure)
    write_audio(base.with_name(base.name + "__neural-48k.wav"), neural)
    return {
        "alias": item["_alias"],
        "bucket": "native",
        "semitones": 0.0,
        "source_wav": str(item["_wav"]),
        "target_wav": str(item["_wav"]),
        "prefix": prefix,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--voicebank", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--native-count", type=int, default=2)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    voicebank = Path(args.voicebank).expanduser().resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = resolve_manifest(voicebank, args.manifest)

    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
    else:
        candidates = [
            root / "control_models" / "neural-waveform-v0.3.0-conditioned-v3-pareto-best.pt",
            root / "control_models" / "neural-waveform-v0.3.0-conditioned-v3-multipitch-best.pt",
            root / "control_models" / "neural-waveform-v0.3.0-conditioned-v3.pt",
            root / "control_models" / "neural-waveform-v0.3.0-conditioned-multipitch-best.pt",
            root / "control_models" / "neural-waveform-v0.3.0-conditioned.pt",
            root / "control_models" / "neural-waveform-v0.3.0-multipitch-best.pt",
            root / "control_models" / "neural-waveform-v0.3.0.pt",
        ]
        checkpoint = next((p for p in candidates if p.is_file()), candidates[-1])
    if not checkpoint.is_file():
        raise RuntimeError(f"neural checkpoint not found: {checkpoint}")

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root / "neural_ab_v0.3.0"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = YuazDDSPResamplerEngine(
        config["yuaz_repo"], config["checkpoint"], output_sr=SAMPLE_RATE,
        registry_path=config.get("registry_path"), ddsp_synthesis_sr=SAMPLE_RATE,
    )
    model, metadata = load_neural_waveform_decoder(checkpoint, device=engine.device)
    prepare_fn = conditioning_function(metadata)

    entries = build_manifest_index(manifest, voicebank)
    _, val_items, val_aliases = alias_split(entries)
    val_pairs, pair_counts = build_pairs(val_items)
    selected = choose_pairs(val_pairs)

    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_role": metadata.get("checkpoint_role"),
        "trainer_generation": metadata.get("trainer_generation"),
        "manifest": str(manifest),
        "held_out_aliases": len(val_aliases),
        "validation_pair_counts": pair_counts,
        "exports": [],
    }

    for idx, item in enumerate(val_items[: max(0, int(args.native_count))], 1):
        prefix = f"native-{idx:02d}__{safe_name(item['_alias'])}"
        report["exports"].append(render_native(engine, model, item, out_dir, prefix, prepare_fn))
        print(f"exported native: {prefix}")

    for name, _, _ in PITCH_BUCKETS:
        pair = selected.get(name)
        if pair is None:
            continue
        prefix = f"{name}__{pair['semitones']:.1f}st__{safe_name(pair['alias'])}"
        report["exports"].append(render_pair(engine, model, pair, out_dir, prefix, prepare_fn))
        print(f"exported {name}: {prefix}")

    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"A/B export complete: {out_dir}")
    print(f"checkpoint: {checkpoint}")
    print(f"trainer generation: {metadata.get('trainer_generation') or 'legacy'}")


if __name__ == "__main__":
    main()
