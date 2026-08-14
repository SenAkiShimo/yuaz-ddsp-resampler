# AI Gender/Formant training — 0.2.8ai.12

This developer pipeline trains a small, bundle-ready `ai_gender_foundation-v1.pt` for `YG`.

## Source

- Canonical dataset: VocalSet: A Singing Voice Dataset, Zenodo record 1442513.
- China-route transport used by the setup script: `Bill13579/vocalset-mirror` through `https://hf-mirror.com`.
- License reported by the mirror: CC BY 4.0. Preserve VocalSet attribution and provenance with distributed derived weights.

The mirror contains parquet shards. `setup-gender-training.command` downloads only those shards and installs developer-only `pyarrow` from the Tsinghua PyPI mirror. No automatic fallback to Hugging Face, PyPI.org, or Zenodo is permitted in this release.

## Training design

Only `straight` singing is used, reducing vocal-technique confounding. Singer identity and vowel are recovered from the original VocalSet filename stored with each audio row. If the mirror does not preserve the original filename or embedded audio bytes, preparation aborts rather than inferring gender from acoustics.

Both sexes are analyzed by the frozen current Yuaz encoder/DDSP decoder. The target is built in Yuaz-native spectral-envelope space:

1. Normalize log spectral-envelope frames to remove broadband level.
2. Group by singer, vowel, and coarse F0 bin.
3. Average within each singer first.
4. Form female and male training-group centroids.
5. Supervise one signed `gender_formant` axis with the male-minus-female centroid direction.

The model never receives a target-singer embedding. Validation holds out whole singers. The pack is hard-limited to `output_scopes=[spectral]`; it cannot modify aperiodicity or harmonic/noise gate.

`YG > 0` is the masculine/lower-formant statistical direction; `YG < 0` is the feminine/higher-formant direction.

## Commands

```bash
./setup-gender-training.command
./train-ai-gender-foundation.command
```

Output:

```text
control_models/ai_gender_foundation-v1.pt
```

A second copy is written to:

```text
~/Documents/Yuaz-DDSP-Backups/control-models/ai_gender_foundation-v1-VocalSet.pt
```

When a compatible pack is present, a later `deep-train-voicebank.command` copies and pins it into the isolated 0.2.8ai.12 generation as `ai_gender_adapter.pt`. The source `.pt` is never modified.
