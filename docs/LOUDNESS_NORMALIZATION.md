# Strict Final-Render Loudness Normalization

The runtime does not normalize from a stored source-WAV gain. It first finishes DDSP rendering, Fidelity refinement, articulation transfer and hybrid assembly. The resulting final waveform is then measured with gated active RMS.

A direct gain is calculated from the actual final active RMS to the configured target. After gain, a soft peak guard only compresses samples near the peak ceiling. Active RMS is measured again and the gain is iteratively corrected until the configured tolerance is reached or the emergency safety bound is encountered.

Default target is -18 dBFS active RMS with 0.05 dB tolerance. The default peak ceiling is -1 dBFS. The emergency 30 dB absolute-gain bound only protects pathological or nearly silent inputs; it is not a normal per-oto normalization limit.

The entire hybrid waveform receives the same normalization operation. Consonant, transition and vowel internal dynamics are not separately leveled.

OpenUtau volume/DYN processing remains downstream.

`.yuaz/loudness.json` stores source oto active-RMS statistics for diagnostics. These source measurements do not determine the runtime gain.
