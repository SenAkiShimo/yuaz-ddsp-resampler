# Changelog

## 0.2.8ai.15

- Updated learned vocal-control scaling.
- Increased deterministic carrier strength for weak control axes.
- Updated YT tension behavior.
- Added ai.15 runtime on port 47887.
- Reuses `.yuaz-0.2.8ai14` as read-only state.
- Removes v0.2.8ai.13 during installation.

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
