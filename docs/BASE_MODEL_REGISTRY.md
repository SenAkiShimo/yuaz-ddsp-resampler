# Yuaz base-model registry

v0.2.8ai.14 can import structurally compatible Yuaz checkpoints instead of relying on a fixed checkpoint filename.

## Commands

- `probe-yuaz-checkpoint.command` checks Encoder, DDSP Decoder, and RVQ tensor coverage and shape compatibility.
- `import-yuaz-checkpoint.command` extracts the tensors required by the resampler and registers a compact local runtime.
- `list-yuaz-checkpoints.command` lists imported models.
- `select-yuaz-checkpoint.command` changes the base model used by the ai.14 runtime.

Imported model data is stored under:

```text
~/Library/Application Support/YuazDDSP/models/
```

A compact runtime retains only `encoder.*`, `ddsp_decoder.*`, and `rvq.*`. Metadata preserves the source checkpoint SHA-256 and training step when available.

## Compatibility

A checkpoint is accepted only when the required resampler modules satisfy the configured coverage thresholds and tensor shapes match the current Yuaz architecture. Additional checkpoint components such as flow generators, temporal predictors, discriminators, optimizers, or scaler state are ignored by the sample-conditioned resampler runtime.

## Voicebank state isolation

Every committed ai.14 generation contains `base_model.json`. Rendering verifies that its recorded source-checkpoint SHA-256 matches the currently selected base model.

A mismatch is treated as incompatible learned state rather than silently reusing adapters or refiners trained against another checkpoint.

v0.2.8ai.14 also does not fall back to `.yuaz-0.2.8ai13` or earlier trained state. Prepare a separate ai.14 generation before rendering a voicebank with ai.14.

## Side-by-side installation

Installing ai.14 does not remove or replace:

- `~/Library/Application Support/YuazDDSP/0.2.8ai.13`
- `~/Library/OpenUtau/Resamplers/Yuaz-DDSP-Resampler-v0.2.8ai.13.sh`
- `~/Library/OpenUtau/Resamplers/Yuaz-DDSP-Resampler-v0.2.8ai.13.yaml`
- `.yuaz-0.2.8ai13` voicebank state directories

v0.2.8ai.14 uses TCP port `47886` and its own OpenUtau wrapper and state namespace.

## Version-specific Deep artifacts

Representative ai.14 learned-state filenames include:

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
