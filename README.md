# Yuaz DDSP Resampler v0.2.7-alpha.1

Experimental sample-conditioned DDSP resampler for UTAU/OpenUtau built around a local Yuaz SGR Encoder/DDSP installation.

This release replaces per-sample voiced articulation transfer with a timbre-neutral canonical articulation dictionary. For multipitch banks, the same `base_alias` is aligned across real UTAU subbanks and its common temporal articulation pattern is retained while broad timbre and spectral tilt are removed.

## What changed

- Dynamic canonical articulation built from real multipitch aliases.
- `prefix.map` / UTAU subbank routing remains authoritative.
- Per-alias fallback uses a timbre-neutral local trajectory when no multipitch counterpart exists.
- Broad spectral tilt is removed from articulation conditioning.
- A 3–9 kHz clarity guard prevents articulation transfer from broadly darkening the DDSP output.
- The single-periodic-source hybrid remains intact: voiced PSOLA is not used.
- Strict final-render active-RMS normalization from v0.2.6-alpha.2 is unchanged.
- Existing Adapter, Anti-Leak, pitch timbre prototypes, Fidelity Refiner, and cache remain compatible.

Prepared banks gain:

```text
.yuaz/
└── articulation/
    ├── index.json
    └── canonical/
        └── *.npz
```

## Upgrade an already adapted bank

Do not reset or Deep Adapt an already trained bank for this release. Run `prepare-voicebank.command` and choose **Fast Profile**. It reuses the existing cache and builds the canonical articulation dictionary without rerunning the Yuaz Encoder or gradient training.

## macOS setup

```bash
cd ~/Downloads/yuaz-ddsp-resampler-v0.2.7-alpha.1
chmod +x *.command scripts/*.command yuaz-ddsp-resampler
./purge-previous-version.command
./setup-macos.command
./configure-macos.command
./self-test.command
```

Then prepare each previously adapted voicebank with Fast Profile and install:

```bash
./install-openutau-macos.command
```

Select `Yuaz-DDSP-Resampler-v0.2.7-alpha.1.sh` after restarting OpenUtau. Default engine port: `47860`.

Use `inspect-voicebank.command` to verify canonical alias count, multipitch canonical count, fallback count, coherence, UTAU subbank routing, and loudness normalization.

## Upstream

Yuaz SGR source code and model weights are not redistributed in this repository. A local upstream checkout and checkpoint are required. See `UPSTREAM.md` and `THIRD_PARTY_NOTICES.md`.
