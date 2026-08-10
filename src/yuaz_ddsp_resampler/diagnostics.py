#!/usr/bin/env python3

import argparse

import json

from pathlib import Path


import numpy as np

import soundfile as sf

import torch


from .core import YuazDDSPResamplerEngine, deterministic_decode, extract_detail_features, extract_f0, read_audio, stable_seed


def load_config(root):

    path = Path(root) / "config.json"

    if not path.exists():

        raise RuntimeError("Run configure-macos.command first.")

    return json.loads(path.read_text(encoding="utf-8"))


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

    audio = read_audio(wav, engine.sr)

    f0_np = extract_f0(audio, engine.sr, engine.hop)

    f0 = torch.from_numpy(f0_np).float().view(1, 1, -1)

    audio_t = torch.from_numpy(audio).float().view(1, 1, -1)

    detail = torch.from_numpy(extract_detail_features(audio, engine.sr, engine.hop)).float().unsqueeze(0)

    with torch.inference_mode():

        z, f0 = engine.encoder(audio_t, f0_override=f0)

    adapter, refiner, record = engine._models_for_input(wav)

    seed = stable_seed(str(wav), "clarity-test")

    base = deterministic_decode(engine.decoder, f0, z, seed, adapter=None, detail=None)

    prototype_index = record.get("subbank_index") if record else None

    adapted = deterministic_decode(engine.decoder, f0, z, seed, adapter=adapter, detail=detail, prototype_index=prototype_index) if adapter is not None else None

    refined = None

    if adapted is not None and refiner is not None:

        adapted_t = torch.from_numpy(adapted).float().view(1, 1, -1)

        with torch.inference_mode():

            refined_t, _ = refiner(adapted_t, detail, f0)

        refined = refined_t[0, 0].cpu().numpy().astype(np.float32)

    out = root / "clarity-test-output"

    out.mkdir(exist_ok=True)

    sf.write(out / "00_original.wav", audio, engine.sr, subtype="PCM_16")

    sf.write(out / "01_base_engine.wav", base, engine.sr, subtype="PCM_16")

    if adapted is not None:

        sf.write(out / "02_clarity_adapter.wav", adapted, engine.sr, subtype="PCM_16")

    if refined is not None:

        sf.write(out / "03_fidelity_refined.wav", refined, engine.sr, subtype="PCM_16")

    print(f"Output: {out}")

    print("Voicebank adapter:", "loaded" if adapter is not None else "not found")

    print("Fidelity refiner:", "loaded" if refiner is not None else "not found")

    if record:

        print("Voicebank ID:", record.get("voicebank_id"))

        print("UTAU subbank:", record.get("subbank_label"), "index", record.get("subbank_index"))


if __name__ == "__main__":

    main()

