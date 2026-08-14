# Multi-pack developer training — 0.2.8ai.13

## Goal

Developer datasets are temporary training dependencies. The deliverables are small `.pt` control packs that can be shipped with Yuaz. End users should not need VocalSet, OSF Phonation Modes, MOCHA-TIMIT, or GTSinger.

## Packs

| Pack | Controls | Training source | DDSP output scope |
|---|---|---|---|
| `ai_control_foundation-v2.pt` | YB/YF/YX/YP | GTSinger Chinese Core | spectral/AP/gate |
| `ai_gender_foundation-v1.pt` | YG | VocalSet | spectral only |
| `ai_phonation_foundation-v1.pt` | YT/YV | OSF Phonation Modes + MOCHA-TIMIT | spectral/AP/gate with orthogonal supervision masks |
| `ai_mouth_foundation-v1.pt` | YO | MOCHA-TIMIT EMA | spectral only |

## Download all remaining data

Run `./setup-training.command` and choose option 1. VocalSet continues to use `~/YuazControlDatasets/VocalSetMirror`, so existing `.part` files from 0.2.8ai.2 are reused. VocalSet offers official Hugging Face or hf-mirror. Phonation Modes is downloaded from the public OSF project (file GUID cwquj). MOCHA-TIMIT is downloaded from the official CSTR repository. All downloaders preserve `.part` files and use HTTP Range where supported.

MOCHA core downloads only the two public speakers (`fsew0`, `msak0`), not a large articulatory corpus.

## Supervision design

- **YT Tension:** OSF Phonation Modes provides singing-domain breathy / modal / pressed examples. Modal is the neutral center; breathy maps negative and pressed positive.
- **YV Voicing/Closure:** MOCHA laryngograph periodicity is an auxiliary glottal-contact/periodicity prior. It trains AP/gate residuals, not the spectral target used by Tension.
- **YO Mouth:** MOCHA EMA upper/lower-lip aperture, normalized within speaker + phone + F0 bin, supervises a spectral-only residual.
- **YG Gender:** VocalSet remains speaker-disjoint and spectral-only.

MOCHA is speech, so it is deliberately an auxiliary prior rather than a singing-style target.

## Train

After all downloads complete, run `./train-all-learned-packs.command`. Existing packs are skipped. The existing GTSinger technique pack is reused rather than retrained.

Then run `./deep-train-ai-voicebank.command` when you want to create a new `.yuaz-0.2.8ai13` generation containing frozen compatible copies of every available pack.
