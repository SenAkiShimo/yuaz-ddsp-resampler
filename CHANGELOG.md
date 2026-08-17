# Changelog

## 0.2.8ai.15

- Removed the second control-magnitude multiplication from learned vocal-control residuals. The neural adapter still receives the requested control value, while the post-network mask now only gates inactive or unvoiced frames.
- Restored stronger deterministic carrier floors underneath learned packs for tension, voicing, gender/formant, mouth, falsetto, mixed voice, and pharyngeal controls so a weak learned residual does not make an axis weaker than its acoustic prior.
- Expanded positive YT tension toward pressed/closed phonation by combining the existing voiced spectral-tilt change with lower aperiodicity and a stronger harmonic gate. Negative YT applies the bounded inverse direction.
- Kept target F0 and source-articulation handling outside the new tension prior.
- Added a dedicated ai.15 runtime identity and TCP port 47887 so v0.2.8ai.14 remains installed for direct A/B comparison.
- Reused compatible v0.2.8ai.14 voicebank generations as a strictly read-only acoustic/training source. ai.15 Prepare and Deep are disabled in this calibration build, preventing writes into `.yuaz-0.2.8ai14`.
- Changed predecessor cleanup to remove v0.2.8ai.13 runtime/OpenUtau entries and `.yuaz-0.2.8ai13` states under the OpenUtau Singers directory while explicitly preserving all v0.2.8ai.14 runtime and trained-state content.

## 0.2.8ai.14

- Added probing, importing, compact runtime extraction, listing, and selection for compatible Yuaz base checkpoints.
- Added source-checkpoint SHA-256 provenance to ai.14 voicebank generations.
- Added render-time rejection of learned state created under a different selected base checkpoint.
- Removed predecessor learned-state fallback from the ai.14 runtime.
- Changed installation to side-by-side operation with v0.2.8ai.13; predecessor purge and state migration are disabled.
- Assigned TCP port 47886 to the ai.14 runtime so v0.2.8ai.13 can remain installed concurrently.
- Added ai.14-specific Deep artifact names and cache directories.
- Added pre-Deep preservation snapshots for installed ai.13 runtime/wrapper and available ai.13 voicebank state.
- Retained the 48 kHz synthesis body, slope-continuity upper-band routing, and output-rate top-band guard from v0.2.8ai.13.
