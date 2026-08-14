# Model weights and provenance

This repository distributes source code only. It does not contain Yuaz base checkpoints, compact runtime checkpoints, voicebank adapters, fidelity refiners, High-Band Foundation weights, or learned-control checkpoints.

## Yuaz base checkpoints

v0.2.8ai.14 accepts structurally compatible Yuaz checkpoints through the local model importer:

```bash
./probe-yuaz-checkpoint.command
./import-yuaz-checkpoint.command
./list-yuaz-checkpoints.command
./select-yuaz-checkpoint.command
```

A full training checkpoint is validated against the current Encoder, DDSP Decoder, and RVQ module shapes. When compatible, the importer writes a compact local runtime containing only those tensors and records:

- source checkpoint filename;
- source checkpoint SHA-256;
- training step when available;
- compact runtime SHA-256.

Imported models are stored outside the repository under the local YuazDDSP application-support directory.

Do not redistribute an upstream checkpoint unless its license or the checkpoint provider explicitly permits redistribution. Source-code licensing does not automatically grant redistribution rights for model weights.

## High-Band Foundation

`train-highband-foundation.command` creates a local checkpoint under `control_models/`. Trained High-Band Foundation weights are derived artifacts and are not committed to this repository.

Before distributing such a checkpoint, review the terms of every dataset represented by the corresponding training audit and shard manifest.

## Learned-control and voicebank weights

Voicebank adapters, Fidelity weights, learned-control packs, and other `.pt` artifacts may have separate dataset or recording provenance. Document and verify redistribution rights independently before publishing them.
