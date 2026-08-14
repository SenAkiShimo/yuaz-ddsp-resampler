# Canonical Articulation

v0.2.7 separates voiced articulation trajectory from subbank timbre.

For each `base_alias`, preparation groups entries after UTAU prefix/suffix stripping. One reliable entry is selected per subbank. Its voiced articulation section is converted into a fixed-length log-spectral trajectory relative to the stable tail. Broad frequency tilt is suppressed and the resulting trajectories are combined with a median across subbanks.

The renderer uses the canonical trajectory when the current oto variant resolves to that `base_alias`. If no canonical file is available, it extracts a neutralized local trajectory from the current sample.

The canonical trajectory controls only magnitude-envelope movement. The target-F0 DDSP waveform remains the sole periodic source; no voiced PSOLA waveform is mixed into the output.

A clarity guard prevents the trajectory from imposing a large broad attenuation over roughly 3–9 kHz. Static brightness and pitch-region identity remain the responsibility of the voicebank/subbank timbre path.
