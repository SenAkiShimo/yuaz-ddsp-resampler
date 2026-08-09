# Changelog

## v0.2.7-alpha.1

- Added canonical articulation dictionaries keyed by UTAU `base_alias`.
- Builds multipitch canonical trajectories from one reliable representative per real subbank.
- Added timbre-neutral trajectory extraction that suppresses broad spectral tilt while retaining temporal local spectral movement.
- Added a 3–9 kHz clarity guard to prevent broad articulation-driven darkening.
- Added neutralized single-sample fallback for aliases without real multipitch counterparts.
- Added exact oto-variant routing by offset/consonant/cutoff so a shared WAV can select the correct `base_alias` and canonical trajectory.
- Added canonical articulation metadata to the global voicebank registry and `.yuaz/articulation/index.json`.
- Preserved single-periodic-source synthesis, Adapter, Anti-Leak, dynamic UTAU timbre routing, Fidelity Refiner, and strict final-render normalization.
- Existing adapted banks only need Fast Profile for this upgrade; no gradient retraining is required.

## v0.2.6-alpha.2

- Replaced bounded source-level loudness compensation with strict final-render active-RMS normalization.
- Restored the v0.2.5-alpha.2 articulation sound chain.
