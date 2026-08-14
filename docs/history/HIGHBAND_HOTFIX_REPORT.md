# High-Band continuity hotfix 1 — engineering report

## Root cause

The pinned High-Band Foundation path and the voicebank source-texture path were mutually exclusive. Once `highband_foundation.pt` existed, runtime used the neural residual and skipped `synthesize_learned_highband()` entirely.

Foundation v1 was trained primarily with sample-domain residual error. For 12–22 kHz singing content, a significant part of the residual is phase-ambiguous or noise-like after a 24 kHz bottleneck. A waveform-heavy objective therefore has a strong incentive to predict near zero where phase cannot be inferred reliably. The result is plausible isolated harmonics/transients but weak temporal occupancy between them.

The Foundation mask begins around 9.5 kHz and reaches full contribution around 12.1 kHz. When the neural branch is sparse, that boundary becomes visible as the observed hard spectral wall.

## Runtime fix

- Keep existing Foundation v1 checkpoints loadable.
- Run the voicebank source-texture branch alongside Foundation when a profile is available.
- Use source texture as a continuity floor rather than a replacement:
  - bridge the edge approximately 8.2–14.6 kHz;
  - compare 12–20 kHz moving RMS between Foundation and the continuity reference;
  - add more continuity only where Foundation is locally weak;
  - retain a small always-on upper-band floor;
  - apply a combined RMS safety limiter.
- Expose before/after temporal-coverage and branch-RMS diagnostics in render logs.

## Foundation v2 training fix

- Train against the same spectrally masked residual used at runtime.
- Reduce waveform loss to 10%.
- Use 56% multi-resolution high-band log-magnitude loss.
- Use 34% framewise band-envelope loss across 9.5–12 / 12–15 / 15–18 / 18–22 kHz.
- Increase receptive field with 8 dilated blocks and 40 hidden channels.
- Output `highband_foundation-v2.pt`; v1 remains a fallback.

## Validation performed

- Python compileall: PASS.
- Bash syntax check for all `.command` files: PASS.
- Full project self-test: PASS.
- Original dead-profile 13–20 kHz regression: 198x upper-band energy restoration.
- Original low-note 16–20 kHz regression: 147.43x upper-band energy restoration.
- Sparse-Foundation continuity regression: PASS.
- Dedicated synthetic temporal-coverage probe: approximately 0.29 before -> 1.00 after hybrid continuity.
- Foundation v2 training objective: finite forward/backward gradients.

## First A/B test

No Deep retraining is required. Existing voicebanks with a pinned v1 `highband_foundation.pt` immediately use the new hybrid runtime after installation. Render the same note at YH100 and run `highband-routing-diagnostic.command`.

If the runtime direction is correct, train Foundation v2 using the already prepared shard manifest, then run `learn-highband.command` for each voicebank.
