# GitHub release setup

Recommended release tag and title:

```text
v0.2.8ai.14
Yuaz DDSP Resampler v0.2.8ai.14
```

## Repository contents

Commit source code, scripts, documentation, lock files, runtime manifests, and preserved source snapshots under `previous_versions/`.

Do not commit generated local state, including:

- `.venv/`;
- `config.json`;
- logs and PID/lock files;
- `.yuaz-*` voicebank state;
- rendered audio;
- base-model or learned-model checkpoints.

Checkpoint files are ignored by default. Publish a model separately only when its provenance and redistribution terms are documented and permit redistribution.

## Release verification

Before tagging a release:

```bash
chmod +x *.command scripts/*.command yuaz-ddsp-resampler
./self-test.command
```

Also run the macOS setup/configuration/doctor path on a clean or isolated runtime when possible.
