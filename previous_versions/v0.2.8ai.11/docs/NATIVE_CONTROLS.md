# Yuaz Native Controls

## YM — Timbre Morph

`YM -100..100` leaves the requested F0 unchanged and shifts only the trained multipitch timbre-routing target. The range maps to `-12..+12` semitones in routing space.

The same YM change is also reflected in the High-Band Bridge. The Adapter's learned relative spectral change between 8 and 12 kHz is measured for the requested YM position and conservatively extended above 12 kHz to color the source high band. No new training is required.

## YD — Learned Detail

`YD -100..100` scales the trained detail-to-latent, detail-to-spectrum, detail-to-aperiodicity, and Fidelity residual paths.

- `YD-100` = 0.0x learned neural detail
- `YD0` = 1.0x
- `YD100` = 1.5x

YD also changes the source high-band contribution. Essential source high-frequency information keeps a conservative floor at negative extremes so consonants are not intentionally erased; positive YD increases the high-band detail contribution moderately.

## YH — High-Band Assist Start

`YH` uses physical frequency units in hundreds of hertz:

- `YH0` = high-band restoration off
- `YH25` = light restoration
- `YH50` = medium restoration
- `YH100` = full restoration; automatic crossover begins below the 12 kHz DDSP Nyquist edge

Above 12 kHz the official 24 kHz Yuaz model cannot contain real spectrum, so the bridge is source-only there. `YH0` is an internal command-line bypass for A/B testing and is not exposed as the normal OpenUtau range.

## High-Band Bridge

The bridge reads the original voicebank WAV at full bandwidth, crops the same OTO region, maps its fixed region and tail to the requested duration, and performs voiced pitch-follow while leaving unvoiced high-frequency consonant texture largely unshifted. Only the high-band component is added back to the Yuaz render; frequencies below the YH boundary remain on the existing Yuaz path.

The bridge requires a source WAV whose native sample rate is above 24 kHz and an output sample rate above 24 kHz. The default output rate remains 44.1 kHz.

## Compatibility

RC3.3 keeps the RC3.2 YM/YD/YH parsing and acoustic control algorithms unchanged. Existing RC3.2 banks should first use `Adopt RC3.2 Baseline`, which copies the current acoustic state without gradient training. The global registry is no longer required for sound identity; voicebank-local immutable state is authoritative.
