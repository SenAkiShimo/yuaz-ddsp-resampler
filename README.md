# Yuaz DDSP Resampler

An external OpenUtau resampler built around Yuaz SGR. The project adds voicebank preparation, multipitch timbre adaptation, articulation preservation, optional learned high-band extension, loudness normalization, and OpenUtau integration.

Chinese documentation: [README.zh-CN.md](README.zh-CN.md)

## Requirements

- macOS
- Python 3
- OpenUtau
- a local Yuaz SGR checkout
- a Yuaz SGR checkpoint

Yuaz SGR source code and model weights are not included in this repository. See [UPSTREAM.md](UPSTREAM.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Setup

```bash
chmod +x *.command scripts/*.command yuaz-ddsp-resampler
./setup-macos.command
./configure-macos.command
```

`setup-macos.command` creates a local `.venv`. It uses the Tsinghua PyPI mirror by default and falls back to the default index if needed.

`configure-macos.command` asks for the local Yuaz SGR repository and checkpoint if they are not found at the default paths.

## Prepare a voicebank

```bash
./prepare-voicebank.command
```

Preparation modes:

1. Fresh Fast Profile
2. Clean Deep Retrain
3. Continue Deep Adapt
4. Relearn High-Band

Training state is stored in:

```text
<voicebank>/.yuaz-alpha8-rc3-2/
```

State-changing preparation creates an external backup before modifying training data. Backups are written under:

```text
~/Documents/Yuaz-DDSP-Backups/<voicebank>/
```

Manual backup and restore:

```bash
./backup-training.command
./restore-previous-training.command
./list-training-backups.command
```

## Install in OpenUtau

```bash
./self-test.command
./install-openutau-macos.command
```

Then restart OpenUtau and select:

```text
Yuaz-DDSP-Resampler-v0.2.7-alpha.8-rc.3.2.sh
```

Uninstall the OpenUtau entry with:

```bash
./uninstall-openutau-macos.command
```

## Controls

| Flag | OpenUtau expression | Range | Default |
|---|---|---:|---:|
| `YM` | Yuaz Timbre Morph | -100..100 | 0 |
| `YD` | Yuaz Learned Detail | -100..100 | 0 |
| `YH` | Yuaz High-Band | 0 or 80..120 | 0 |

`YH0` disables the learned high-band extension. Non-zero `YH` values are interpreted as the Yuaz-only crossover in hundreds of hertz.

More details are in [docs/CONTROLS.md](docs/CONTROLS.md).

## Voicebank inspection

```bash
./inspect-voicebank.command
```

## Loudness settings

```bash
./loudness-settings.command
```

## License

Source code in this repository is provided under the MIT License. Yuaz SGR code, checkpoints, datasets, and other third-party components have their own terms.
