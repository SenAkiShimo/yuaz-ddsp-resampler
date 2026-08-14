# Changelog

## 0.2.8ai.13

- Added a wide 8.2–13.8 kHz slope-continuous full-band crossover.
- Added an output-rate-aware harmonic ceiling and separate harmonic/aperiodic terminal tapers.
- Added a post-refinement terminal guard for 44.1 kHz output.
- Added band-specific bridge, seam, presence, air, and top-band level control.
- Kept the 0.2.8ai.12 upper-band path and earlier full-band mixer as runtime fallbacks.
- Added migration from `.yuaz-0.2.8ai12` to `.yuaz-0.2.8ai13`.
- Added regression coverage for upper-band slope continuity and terminal-band suppression.

## 0.2.8ai.12

- Added a frequency-dependent upper-band parameter head for 48 kHz synthesis.
- Added band-wise upper-band safety control instead of a single global high-band limiter.
- Extended spectral envelope and aperiodicity above the 24 kHz analysis band.

## 0.2.8ai.11

- Added the dual-rate 24 kHz analysis / 48 kHz synthesis path.
- Added High-Band Foundation v2 compatibility and source-texture continuity refinement.
- Added validated voicebank-state migration into the ai.11 namespace.

Earlier implementation notes are preserved in `previous_versions/` and `docs/history/`.
