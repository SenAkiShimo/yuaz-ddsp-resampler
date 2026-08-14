# Voicebank adaptation — RC3.3

RC3.3 keeps RC3.2 Stage A and Stage B unchanged, then optionally adds a conservative Stage C Fidelity residual layer in Deep modes.

## Stage A — Identity

Learns singer identity, multipitch routing, selective Anti-Leak and Canonical Articulation using the RC3.2 strategy.

## Stage B — Clarity

Freezes content scrubber, latent identity, global/pitch timbre codes and timbre-routing projections. Only the same RC3.2 spectral/AP/detail subset is calibrated. The objectives remain source-relative body/presence balance, low-mid excess guard, F0-aware harmonic peak/valley contrast, inter-harmonic valley fill and stable-voiced AP correction. Validation keeps Stage A when Stage B is worse.

## Stage C — Fidelity residual

Deep modes train a small residual refiner against the completed Stage-A/B output. Its hard residual RMS ratio is limited to 0.085. Stage C does not replace High-Band v3 or Canonical Articulation.

## Transactional generations

Training never edits the active generation in place. New state is built in `.staging-*`, validated and SHA-256 pinned, then `ACTIVE.json` is atomically switched. A failed job leaves the previous sound active.

`Adopt RC3.2 Baseline` performs no gradient training and is the recommended first migration for an existing RC3.2 bank.
