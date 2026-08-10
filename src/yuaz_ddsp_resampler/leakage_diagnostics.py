#!/usr/bin/env python3

import argparse

import json

from pathlib import Path


import numpy as np

import torch

import torch.nn.functional as F


from .core import YuazDDSPResamplerEngine, extract_f0, read_audio, stable_seed

from .prepare import timbre_perturb_audio


def load_config(root):

    path = Path(root) / "config.json"

    if not path.exists():

        raise RuntimeError("Run configure-macos.command first.")

    return json.loads(path.read_text(encoding="utf-8"))


def normalize_content(x):

    return F.layer_norm(x.transpose(1, 2), (x.shape[1],)).transpose(1, 2)


def distance(a, b):

    n = min(a.shape[-1], b.shape[-1])

    a = normalize_content(a[..., :n])

    b = normalize_content(b[..., :n])

    return float(F.smooth_l1_loss(a, b).detach())


def encode(engine, audio, f0_t):

    audio_t = torch.from_numpy(audio).float().view(1, 1, -1)

    with torch.inference_mode():

        z, _ = engine.encoder(audio_t, f0_override=f0_t)

    return z


def cross_subbank_probe(adapter, record, wav):

    if not record or not record.get("voicebank_root") or not record.get("base_alias"):

        return None

    root = Path(record["voicebank_root"])

    manifest_path = root / ".yuaz-alpha8-rc3-2" / "manifest.json"

    if not manifest_path.exists():

        return None

    try:

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    except Exception:

        return None

    candidates = []

    for item in manifest.get("entries", []):

        if item.get("status") == "error" or not item.get("cache"):

            continue

        if item.get("base_alias") != record.get("base_alias"):

            continue

        if item.get("subbank_index") == record.get("subbank_index"):

            continue

        candidates.append(item)

    if not candidates:

        return None

    ref = None

    for item in manifest.get("entries", []):

        try:

            if Path(item.get("wav_path", "")).resolve() == Path(wav).resolve() and item.get("cache"):

                ref = item

                break

        except Exception:

            continue

    if ref is None:

        return None

    ref_data = np.load(root / ref["cache"], allow_pickle=False)

    z0 = torch.from_numpy(ref_data["latent"].astype(np.float32)).unsqueeze(0)

    raw_distances = []

    scrubbed_distances = []

    rows = []

    with torch.inference_mode():

        c0 = adapter.content_representation(z0)

        for item in candidates[:8]:

            data = np.load(root / item["cache"], allow_pickle=False)

            z1 = torch.from_numpy(data["latent"].astype(np.float32)).unsqueeze(0)

            frames = z1.shape[-1]

            z0i = F.interpolate(z0, size=frames, mode="linear", align_corners=False)

            c0i = F.interpolate(c0, size=frames, mode="linear", align_corners=False)

            c1 = adapter.content_representation(z1)

            rd = distance(z0i, z1)

            sd = distance(c0i, c1)

            raw_distances.append(rd)

            scrubbed_distances.append(sd)

            rows.append({

                "subbank": item.get("subbank_label"),

                "raw_distance": rd,

                "scrubbed_distance": sd,

            })

    raw = float(np.mean(raw_distances)) if raw_distances else 0.0

    scrubbed = float(np.mean(scrubbed_distances)) if scrubbed_distances else 0.0

    reduction = 0.0 if raw <= 1e-8 else 100.0 * (1.0 - scrubbed / raw)

    return {

        "base_alias": record.get("base_alias"),

        "comparison_count": len(rows),

        "raw_cross_subbank_distance_mean": raw,

        "scrubbed_cross_subbank_distance_mean": scrubbed,

        "relative_reduction_percent": reduction,

        "comparisons": rows,

    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("wav")

    parser.add_argument("--project-root", required=True)

    args = parser.parse_args()

    root = Path(args.project_root).resolve()

    wav = Path(args.wav).expanduser().resolve()

    config = load_config(root)

    engine = YuazDDSPResamplerEngine(

        config["yuaz_repo"], config["checkpoint"],

        transition_ms=config.get("transition_ms", 70), use_rvq=config.get("use_rvq", False),

        output_sr=config.get("output_sr", 44100), registry_path=config.get("registry_path"),

    )

    adapter, refiner, record = engine._models_for_input(wav)

    if adapter is None:

        raise RuntimeError("No prepared Anti-Leak adapter was found for this WAV.")


    audio = read_audio(wav, engine.sr)

    f0_np = extract_f0(audio, engine.sr, engine.hop)

    f0_t = torch.from_numpy(f0_np).float().view(1, 1, -1)

    z0 = encode(engine, audio, f0_t)


    raw_distances = []

    scrubbed_distances = []

    with torch.inference_mode():

        c0 = adapter.content_representation(z0)

        for i in range(16):

            aug = timbre_perturb_audio(audio, engine.sr, stable_seed(str(wav), "leakage-probe-v2", i))

            za = encode(engine, aug, f0_t)

            raw_distances.append(distance(z0, za))

            ca = adapter.content_representation(za)

            scrubbed_distances.append(distance(c0, ca))


    raw = float(np.mean(raw_distances))

    scrubbed = float(np.mean(scrubbed_distances))

    reduction = 0.0 if raw <= 1e-8 else 100.0 * (1.0 - scrubbed / raw)

    result = {

        "wav": str(wav),

        "voicebank_id": record.get("voicebank_id") if record else None,

        "utau_subbank": record.get("subbank_label") if record else None,

        "utau_subbank_index": record.get("subbank_index") if record else None,

        "base_alias": record.get("base_alias") if record else None,

        "perturbation_probe_count": 16,

        "raw_latent_timbre_sensitivity_mean": raw,

        "raw_latent_timbre_sensitivity_std": float(np.std(raw_distances)),

        "scrubbed_content_timbre_sensitivity_mean": scrubbed,

        "scrubbed_content_timbre_sensitivity_std": float(np.std(scrubbed_distances)),

        "relative_reduction_percent": reduction,

        "cross_subbank_same_alias_probe": cross_subbank_probe(adapter, record, wav),

        "interpretation": "lower scrubbed sensitivity is better; cross-subbank same-alias comparisons are more representative when the bank has real multipitch recordings",

    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":

    main()

