# Architecture

The runtime resolves the current UTAU oto entry and subbank, analyzes the source with Yuaz SGR, applies the prepared voicebank adapter, applies articulation-envelope motion, synthesizes the target-F0 DDSP signal, optionally adds learned high-band extension, and performs final loudness normalization.

The periodic portion of the rendered note remains DDSP-generated after reliable voiced onset. Unvoiced source material may be preserved around consonant and transient regions.

Voicebank preparation stores per-bank state outside the source WAV files. The active state directory for this release is `.yuaz-alpha8-rc3-2`.
