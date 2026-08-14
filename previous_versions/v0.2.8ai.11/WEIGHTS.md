# Weight provenance

This source package does not contain a High-Band Foundation checkpoint.

`train-highband-foundation.command` now creates:

```text
control_models/highband_foundation-v2.pt
```

The runtime remains compatible with an existing v1 `highband_foundation.pt`, so the continuity hotfix can be tested without retraining first.

A trained checkpoint is a derived artifact of the exact source records listed by:

```text
~/YuazControlDatasets/HighBandFoundation/audit.json
~/YuazControlDatasets/HighBandFoundation/shards/manifest.json
```

Before distributing a PT, review the licenses/terms of every dataset that contributed accepted training shards. Do not assume the source-code license automatically applies to derived model weights.

The existing learned-control PTs have their own dataset provenance and should be documented separately in a public release.
