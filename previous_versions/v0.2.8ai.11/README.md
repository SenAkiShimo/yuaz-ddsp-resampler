# Yuaz DDSP Resampler 0.2.8ai.11

## 0.2.8ai.11 — dual-rate 48 kHz DDSP body

0.2.8ai.11 keeps Yuaz analysis/latent conditioning at 24 kHz for checkpoint and voicebank-state compatibility, but runs the harmonic/noise DDSP synthesis body at 48 kHz. A complementary 9–12.1 kHz crossover preserves the trained low/mid body while allowing YH0 itself to contain real upper-band DDSP energy. YH remains an optional voicebank/Foundation refinement layer. Existing 0.2.8ai.10 adapter, Fidelity, articulation, high-band profile, and pinned Foundation state are migrated; derived caches are rebuilt.


Experimental High-Band Foundation v1 branch.

This release freezes the 0.2.8ai.9 acoustic/control behavior and adds a trainable bandwidth-extension model instead of continuing to hand-synthesize upper harmonics. The foundation learns paired full-band singing versus the same audio passed through a 24 kHz bottleneck. Runtime output is hard-masked to the upper band so the model cannot rewrite the main Yuaz body.

Workflow:

```bash
./setup-macos.command
./configure-macos.command
./self-test.command
./install-openutau-macos.command
./doctor.command

./audit-highband-datasets.command
./prepare-highband-training.command
./train-highband-foundation.command
./probe-highband-foundation.command
./learn-highband.command
```

The audit uses existing local GTSinger, VocalSet and Phonation Modes data and does not download new datasets. Low-F0 material is oversampled and evaluated separately.

Output model:

```text
control_models/highband_foundation-v1.pt
```

The model is pinned into a voicebank generation as `highband_foundation.pt`. If no foundation is available, YH falls back to the 0.2.8ai.9 voicebank-profile high-band path.

See `README.zh-CN.md` for the full workflow and design notes.
## High-Band continuity hotfix

This package fixes the sparse YH100 upper-band failure that could appear after a v1 Foundation was pinned. Existing v1 checkpoints remain compatible: the Foundation now supplies learned events while the voicebank source-texture path provides an adaptive 8–20 kHz continuity floor. Foundation v2 training uses phase-tolerant multi-resolution magnitude and framewise band-envelope losses instead of a waveform-dominated objective. No Deep retraining is required for the first test with an existing v1 checkpoint.

Use `highband-routing-diagnostic.command` after a YH render to inspect temporal coverage before/after the continuity assist.

