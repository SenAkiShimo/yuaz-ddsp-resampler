# Vocal Controls — 0.2.8ai.11

Yuaz exposes exactly 12 custom controls:

- YM Timbre Morph
- YD Learned Detail
- YH High-Band
- YT Tension
- YB Breathiness
- YV Voicing
- YG Gender / Formant
- YO Mouth / Resonance
- YF Falsetto
- YX Mixed Voice
- YP Pharyngeal

YA Attack is retired and ignored.

## Modular learned ownership

A learned control pack declares exactly which axes it owns. The deterministic DDSP implementation is bypassed only for those axes.

Current technique pack: YB positive direction, YF, YX, YP.

New Gender pack: YG, signed, `output_scopes=[spectral]`. When present, deterministic formant warp is bypassed for YG and the learned pack operates on the Yuaz spectral envelope. It is structurally unable to change AP or harmonic/noise gate.

If a compatible pack is absent or rejected by checkpoint/provenance validation, the deterministic implementation remains available.

- `YR` Raw WAV Bypass: `0` normal Yuaz rendering, `1` source-WAV passthrough for breath/noise/special samples.
