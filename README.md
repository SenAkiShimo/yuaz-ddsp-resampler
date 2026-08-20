# Yuaz DDSP Resampler v0.2.9

A sample-conditioned Yuaz/DDSP resampler for OpenUtau on macOS.

v0.2.9 promotes the validated vocal-control runtime, including clearer YF/YB separation and F0-relative falsetto register shaping, while keeping the existing 48 kHz synthesis body and ai.14 voicebank state as read-only compatibility data.

## Highlights

- Explicit target-F0-conditioned DDSP resynthesis with source-preserving articulation handling.
- Voicebank-aware adapters when compatible ai.14 state is available, with safe base-render fallback otherwise.
- Vocal controls for tension, breathiness, voicing, gender/formant, mouth/resonance, falsetto, mixed voice, and pharyngeal shaping.
- F0-relative falsetto register shaping designed to remain distinct from breathiness.
- 48 kHz synthesis body with upper-band continuity routing and terminal guard.
- Production runtime on TCP port `47888`.
- Reuses `.yuaz-0.2.8ai14` as read-only compatibility state; v0.2.9 does not retrain or overwrite it.

## Commands

Human-facing commands use one launcher. Implementations remain under `scripts/`.

```bash
./commands/run.command list
./commands/run.command find yf
./commands/run.command doctor
```

## Model weights

Model checkpoints are **not included in this repository**. Obtain a compatible Yuaz checkpoint from its authorized source and import it locally:

```bash
./commands/run.command probe-yuaz-checkpoint
./commands/run.command import-yuaz-checkpoint
./commands/run.command list-yuaz-checkpoints
./commands/run.command select-yuaz-checkpoint
```

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

The production OpenUtau wrapper is `Yuaz-DDSP-Resampler-v0.2.9.sh`.

## Clean up older Yuaz resamplers

After installing v0.2.9, older Yuaz runtime directories and OpenUtau wrappers can be removed with:

```bash
./commands/run.command cleanup-legacy-yuaz-resamplers
```

The cleanup command preserves voicebank `.yuaz-*` state, checkpoints, shared environments, and source repositories.

## Voicebank state

v0.2.9 reads compatible `.yuaz-0.2.8ai14` learned state without modifying it. If no compatible learned state can be resolved, the runtime falls back to the base source-conditioned rendering path instead of failing the OpenUtau render.

## Upstream

Yuaz DDSP Resampler uses the Yuaz SGR encoder and DDSP decoder architecture. The upstream project is documented in [`UPSTREAM.md`](UPSTREAM.md).
