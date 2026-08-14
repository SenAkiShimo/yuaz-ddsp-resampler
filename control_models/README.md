# Modular AI control packs

0.2.8ai.13 supports independent Yuaz-native DDSP control packs. Stable real weights can later be bundled in `control_models/`, so end users do not need any developer datasets.

- `ai_control_foundation-v2.pt` — existing GTSinger Chinese Core: YB Breathiness, YF Falsetto, YX Mixed Voice, YP Pharyngeal.
- `ai_gender_foundation-v1.pt` — VocalSet: signed YG Gender/Formant, spectral-envelope only.
- `ai_phonation_foundation-v1.pt` — OSF Phonation Modes + MOCHA-TIMIT: signed YT Tension + YV Voicing/Closure.
- `ai_mouth_foundation-v1.pt` — MOCHA-TIMIT EMA: signed YO Mouth/Resonance, spectral-envelope only.

A Deep generation pins compatible packs by copying them. Source models and predecessor generations are never modified. No synthetic smoke-test weight belongs in a release package.
