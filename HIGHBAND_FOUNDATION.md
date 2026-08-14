# High-Band Foundation v2 / v1-compatible runtime

## Runtime integration

The original v1 checkpoint is still accepted. The runtime no longer lets a pinned Foundation completely replace the voicebank source-texture path. Instead:

```text
Yuaz 24 kHz body
  + Foundation residual (predictable upper-band events)
  + voicebank source-texture continuity floor (8–20 kHz bridge/gap fill)
  -> safety limiter -> output
```

This fixes the failure mode where a Foundation checkpoint produced a few convincing >12 kHz events but left most voiced frames nearly empty, making the 9–12 kHz boundary visible as a hard spectral wall.

The continuity branch is not a second vocal waveform. It is derived from the already-rendered Yuaz signal and constrained by the voicebank high-band profile. It is used strongly in the 8.2–14.6 kHz bridge and adaptively above 12 kHz only when the Foundation branch is locally weak.

## Foundation v2 training objective

Pair construction is unchanged:

- target: accepted full-band singing at 48 kHz;
- input: the same target through a 24 kHz bottleneck and back to 48 kHz;
- target residual: `target - input`, masked with the same runtime upper-band mask.

The v1 objective was dominated by sample-domain residual error. Stochastic/phase-ambiguous high-band energy is not uniquely predictable from a 24 kHz bottleneck, so a waveform-heavy objective can minimize error by collapsing those components toward zero.

V2 uses:

- 10% smooth waveform residual loss;
- 56% multi-resolution high-band log-magnitude loss;
- 34% framewise band-envelope loss over 9.5–12, 12–15, 15–18 and 18–22 kHz.

The v2 model also uses a wider dilated receptive field. Existing v1 weights remain loadable and immediately benefit from the runtime continuity fix.

## Runtime bands

The Foundation residual remains masked with the original approximately 9.5 kHz crossover, full contribution near 12.1 kHz and upper limit near 22 kHz. The hybrid continuity bridge deliberately begins lower, around 8.2 kHz, so the output does not expose the Foundation mask boundary as a visible wall.

## Validation

`self-test.command` includes a sparse-Foundation regression. A deliberately intermittent Foundation branch is combined with the continuity branch and must substantially improve upper-band temporal coverage rather than merely increase isolated peaks.

After an actual YH render, run:

```bash
./highband-routing-diagnostic.command
```

The result includes `highband_temporal_coverage_before`, `highband_temporal_coverage_after`, branch RMS values and whether the hybrid continuity path was used.

## Distribution

Do not assume the PT shares the source-code license. Derived model rights follow the datasets that entered the training shards. Preserve audit/shard provenance and review dataset terms before publishing weights.
