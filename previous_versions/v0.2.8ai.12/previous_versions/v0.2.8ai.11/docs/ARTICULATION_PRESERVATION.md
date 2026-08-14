# Articulation preservation

The resampler preserves UTAU articulation without preserving a second voiced source waveform.

Before reliable periodic onset, original unvoiced consonant/transient audio can remain in the output. After voiced onset, the target-F0 DDSP waveform is the only periodic source.

v0.2.7 changes voiced articulation conditioning from a single source recording to a canonical trajectory. The preparer groups the same UTAU `base_alias` across real subbanks and extracts the temporal spectral motion common to those recordings. Broad static spectral tilt is suppressed so subbank timbre remains the responsibility of the timbre path rather than the articulation path.

When no real multipitch counterpart exists, the current sample is converted into a timbre-neutral local trajectory instead of applying its absolute spectral envelope.

This design aims to preserve how a bank moves into a vowel while reducing articulation-time timbre leakage and broad high-frequency darkening.
