# Yuaz DDSP Resampler v0.2.8ai.14

A sample-conditioned Yuaz/DDSP resampler for OpenUtau on macOS.

v0.2.8ai.14 adds a base-model registry for compatible Yuaz checkpoints while retaining the 48 kHz synthesis body, upper-band continuity routing, and output-rate top-band guard introduced in the preceding release.

## Highlights

- Imports structurally compatible Yuaz `.pt` checkpoints without relying on a fixed filename.
- Extracts a compact runtime checkpoint containing only the Encoder, DDSP Decoder, and RVQ tensors required by the resampler.
- Records source-checkpoint SHA-256 and training-step provenance.
- Isolates learned voicebank state by base checkpoint and rejects mismatched state at render time.
- Installs side by side with v0.2.8ai.13 instead of replacing it.
- Uses a separate runtime port, OpenUtau wrapper, state namespace, trained-artifact filenames, and cache directories.
- Keeps destructive predecessor purge disabled in this release.

See [`docs/BASE_MODEL_REGISTRY.md`](docs/BASE_MODEL_REGISTRY.md) and [`docs/SIDE_BY_SIDE_SAFETY.md`](docs/SIDE_BY_SIDE_SAFETY.md) for details.

## Commands

Human-facing commands use one launcher. Implementations remain under `scripts/`.

```bash
./commands/run.command list
./commands/run.command find yv
./commands/run.command doctor
```

See [`commands/README.md`](commands/README.md) for usage and aliases.

## Model weights

Model checkpoints are **not included in this repository**. Obtain a compatible Yuaz checkpoint from its authorized source and import it locally:

```bash
./commands/run.command probe-yuaz-checkpoint
./commands/run.command import-yuaz-checkpoint
./commands/run.command list-yuaz-checkpoints
./commands/run.command select-yuaz-checkpoint
```

The importer validates Encoder / DDSP Decoder / RVQ coverage before registering a model. Full training checkpoints may contain additional generator, discriminator, optimizer, and scaler state; these components are not required by the OpenUtau resampler runtime.

See [`WEIGHTS.md`](WEIGHTS.md) for redistribution and provenance notes.

## Install

```bash
chmod +x commands/run.command yuaz-ddsp-resampler
./commands/run.command setup-macos
./commands/run.command configure-macos
./commands/run.command self-test
./commands/run.command install-openutau-macos
./commands/run.command doctor
```

During configuration, provide either a full compatible Yuaz checkpoint or a compact runtime checkpoint previously produced by the importer.

## Voicebank preparation

```bash
./commands/run.command deep-train-voicebank
```

v0.2.8ai.14 writes prepared state only under `.yuaz-0.2.8ai14`. It does not migrate, rename, overwrite, or delete `.yuaz-0.2.8ai13` state.

Version-specific learned artifacts include:

```text
adapter.ai14.pt
timbre_profiles.ai14.pt
training.ai14.json
fidelity_refiner.ai14.pt
fidelity_training.ai14.json
deep_validation.ai14.json
highband_profiles_v3.ai14.json
cache_ai14/
highband_cache_v3_ai14/
```

## OpenUtau coexistence

v0.2.8ai.14 uses TCP port `47886`; v0.2.8ai.13 remains on its existing port. Both resampler wrappers can remain installed at the same time.

`purge-previous-version` is intentionally disabled in v0.2.8ai.14.

## Upstream

Yuaz DDSP Resampler uses the Yuaz SGR encoder and DDSP decoder architecture. The upstream project is documented in [`UPSTREAM.md`](UPSTREAM.md).
