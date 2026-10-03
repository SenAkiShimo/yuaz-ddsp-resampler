# v0.4 V3 / V4 / HiFi blind listening test

This diagnostic compares three voicebank-matched checkpoints under exactly the same source OTO,
target-F0 trajectory, decoder seed, and output sample rate:

- `v3_robust`: v0.4 robust synthetic-pair V3 Pareto checkpoint.
- `v4_robust`: v0.4 robust source-detail V4 Pareto checkpoint.
- `hifi`: v0.4 HiFi Pareto checkpoint with gated 6-22 kHz source texture.

The exporter is intended for single-subbank voicebanks where recorded same-alias cross-pitch pairs
are sparse or absent.

## Run

```bash
./commands/run.command export-v04-listening-ab "/path/to/voicebank"
```

Defaults:

- 3 held-out voiced OTO entries.
- 0, +5, +9, and +14 semitone targets.
- 48 kHz PCM-24 export.
- deterministic A/B/C shuffling per item and pitch shift.

Custom example:

```bash
./commands/run.command export-v04-listening-ab "/path/to/voicebank" \
  --samples 5 \
  --shifts "0,3,7,12,14"
```

## Required checkpoints

The command intentionally refuses to run unless all three checkpoints exist under the same
voicebank-scoped v0.4 directory:

```text
control_models/v0.4/<voicebank-id>/v3/neural-waveform-v0.4-v3-pareto-best.pt
control_models/v0.4/<voicebank-id>/v4/neural-waveform-v0.4-v4-pareto-best.pt
control_models/v0.4/<voicebank-id>/hifi/neural-waveform-v0.4-hifi-pareto-best.pt
```

Every checkpoint's `metadata.voicebank` and trainer generation are verified before rendering.

## Output

The default output folder is:

```text
listening_tests/<voicebank-id>-v04/
```

It contains:

- `blind/`: A/B/C files to judge first.
- `labeled/`: the same renders with model names exposed.
- `reference/`: source original, diagnostic librosa pitch-shift reference, and V4 DDSP structure.
- `scorecard.csv`: clarity/high-band ratings and notes.
- `blind_key.json`: reveal key; do not open until scoring is complete.
- `report.json`: checkpoint provenance and sample metadata.

The librosa pitch-shifted waveform is diagnostic only and is not a ground-truth target.

## Decision rule

Judge articulation and high-band fidelity separately.

If HiFi improves openness/high-band detail without reducing articulation compared with robust V4,
the HiFi route is suitable for consolidation.

If V4 and HiFi remain similarly blurred in consonant edges or CV transitions, especially while
high-band scores improve, the next experiment should target onset/articulation time structure rather
than adding more broad high-frequency energy.
