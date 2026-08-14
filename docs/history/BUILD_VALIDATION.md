# Build validation — 0.2.8ai.13

Validated before packaging:

- Python `compileall` passes for the current `src/` tree.
- `bash -n` passes for all root and `scripts/` command files.
- Full project self-test passes while retaining the existing pitch, articulation, high-band, Foundation v1/v2, ai.11 and ai.12 regressions.
- The complete ai.12 source tree bundled at `previous_versions/v0.2.8ai.12/` matches the source package used to build ai.13 file-for-file.
- ai.13 synthetic voiced-body regression preserves 7–10 kHz energy at about 1.000× versus ai.12 while keeping 15–18 kHz materially above the ai.11 fallback.
- ai.13 terminal regression reduces 20–22 kHz energy to about 0.007× of the ai.12 test path, without removing the useful 12–18 kHz body.
- ai.13 slope-continuity regression produces a progressive 9–15 kHz decline rather than a 10–12 kHz roof followed by a density cliff.
- Migration simulation verifies ai.12 → ai.13 state migration, Foundation/adapter preservation, and derived-cache omission.
- No newly trained `.pt` checkpoint is required or bundled by this release.

The release is a runtime synthesis/crossover correction. Existing Deep/Quick/Fidelity/Foundation training is intentionally reusable.
