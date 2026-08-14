# GitHub release setup

Recommended repository name:

```text
yuaz-ddsp-resampler
```

Recommended release tag and title:

```text
v0.2.8ai.13
Yuaz DDSP Resampler v0.2.8ai.13
```

## Public repository contents

Commit source code, scripts, documentation, lock files, manifests required by the runtime, and preserved source snapshots under `previous_versions/`.

Do not commit generated local state such as `.venv`, `config.json`, logs, voicebank `.yuaz-*` directories, rendered WAV files, or global registry caches.

Trained checkpoints are ignored by default. Publish weights separately only when their provenance and redistribution terms are documented.

## Suggested first push

```bash
git init
git add .
git commit -m "Release v0.2.8ai.13"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

Create the release from tag `v0.2.8ai.13` after CI and local self-tests pass.
