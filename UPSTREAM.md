# Upstream relationship

Yuaz DDSP Resampler is built around the encoder and DDSP decoder provided by Yuaz SGR:

- Upstream repository: https://github.com/Yuaz-Club/yuaz-sgr

The resampler is intentionally maintained as a separate repository because its runtime interface, processing pipeline, release cadence, and planned voicebank adaptation system differ substantially from the upstream singing-synthesis project.

The public repository does not vendor the Yuaz SGR source tree or checkpoint. Users configure a local checkout instead.

## Git remote

Maintainers can keep an explicit local reference to the upstream repository without making this project a GitHub fork:

```bash
git remote add yuaz-upstream https://github.com/Yuaz-Club/yuaz-sgr.git
git remote -v
```

This does not modify the upstream repository.

## Optional submodule

A future release can pin a specific upstream commit without copying its files:

```bash
git submodule add https://github.com/Yuaz-Club/yuaz-sgr.git upstream/yuaz-sgr
```

The submodule keeps the histories separate and records the upstream commit used by the resampler.
