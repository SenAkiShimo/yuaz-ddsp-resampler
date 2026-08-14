# Build validation — 0.2.8ai.11 High-Band continuity hotfix 1

Validated before packaging:

- Python `compileall` passes for `src/`.
- `bash -n` passes for all root and `scripts/` command files.
- Full project self-test passes, including the original high-band regressions and the new sparse-Foundation temporal-continuity regression.
- The synthetic sparse-Foundation test increased upper-band temporal coverage from about 0.29 to 1.00 with the hybrid continuity floor.
- Foundation v1 checkpoint loading remains compatible.
- Foundation v2 training loss is finite and backpropagates through the wider model.
- Foundation v2 has 92,681 parameters.
- The package does not contain a trained/test `.pt` checkpoint.

The important runtime change is usable immediately with an existing pinned v1 `highband_foundation.pt`; retraining v2 is optional for the first A/B test.
