# Yuaz DDSP Resampler

Yuaz DDSP Resampler is an OpenUtau resampler built around a dual-rate DDSP pipeline. Version `0.2.8ai.13` uses 24 kHz analysis/latent features and a 48 kHz synthesis body, with a slope-continuous upper-band crossover and an output-rate-aware terminal guard.

Chinese documentation: [README.zh-CN.md](README.zh-CN.md)

## Features

- OpenUtau-compatible resampler wrapper for macOS.
- 24 kHz analysis with 48 kHz DDSP synthesis.
- Frequency-dependent upper-band spectral envelope, aperiodicity, and harmonic/noise mixing.
- Wide 8.2–13.8 kHz crossover for smooth transition across the analysis-band edge.
- Output-rate-aware harmonic ceiling and terminal filtering for 44.1 kHz output.
- Optional High-Band Foundation refinement and voicebank-specific high-band profiles.
- Voicebank adapter, Fidelity Refiner, articulation preservation, loudness normalization, and learned vocal-control packs.
- Transactional voicebank-state migration with rollback support.

## Requirements

- macOS on Apple Silicon
- Python 3.14
- OpenUtau

The project uses a pinned Python environment. Exact package versions are listed in `requirements.lock.txt`.

## Installation

```bash
cd yuaz-ddsp-resampler-v0.2.8ai.13
chmod +x *.command scripts/*.command yuaz-ddsp-resampler

./setup-macos.command
./configure-macos.command
./self-test.command
./install-openutau-macos.command
./doctor.command
```

The installer validates and migrates compatible voicebank state before removing older installed Yuaz runtime/wrapper/state containers. Source WAV/OTO files, training datasets, and `~/Documents/Yuaz-DDSP-Backups` are not removed.

## Diagnostics

After rendering at least one note in OpenUtau:

```bash
./highband-routing-diagnostic.command
```

The current full-band backend is:

```text
dual-rate-48k-ddsp-body-v3-slope-continuity-topguard
```

For additional diagnostics, see the `*-test.command` and `*-diagnostic.command` scripts in the repository root.

## Training and voicebank preparation

Existing prepared voicebanks can reuse compatible adapter, Fidelity, articulation, high-band profile, and Foundation state. Training utilities are included for users who need to prepare or retrain these components.

Relevant documentation:

- `docs/ARCHITECTURE.md`
- `docs/VOICEBANK_ADAPTATION.md`
- `docs/ARTICULATION_PRESERVATION.md`
- `docs/LEARNED_HIGHBAND.md`
- `HIGHBAND_FOUNDATION.md`
- `WEIGHTS.md`

## Repository layout

```text
src/yuaz_ddsp_resampler/   Python runtime and DSP implementation
scripts/                   macOS setup, migration, diagnostics, and training tools
control_models/            model metadata/documentation; trained weights are not committed
docs/                      architecture and developer documentation
previous_versions/         preserved source snapshots for compatibility/reference
```

Historical engineering notes and build manifests are stored in `docs/history/` and are not part of the runtime path.

## Development

Run the local checks before committing:

```bash
python3 -m compileall -q src/yuaz_ddsp_resampler
./self-test.command

for f in *.command scripts/*.command; do
  bash -n "$f"
done
```

See `CONTRIBUTING.md` and `docs/GITHUB_SETUP.md` for repository and release guidance.

## Weights and datasets

Trained checkpoints are intentionally excluded from Git. Dataset and derived-weight licensing may differ from the source-code license. See `WEIGHTS.md` and `THIRD_PARTY_NOTICES.md` before redistributing weights or datasets.

## License

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
