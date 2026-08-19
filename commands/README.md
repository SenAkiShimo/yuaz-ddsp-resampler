# Yuaz command launcher

`commands/run.command` is the public entry point for Yuaz maintenance, setup, training, and diagnostics.

The actual implementations live in `scripts/`. Keeping implementation scripts there avoids dozens of duplicate `.command` wrappers in the repository root.

## Usage

```bash
./commands/run.command <name> [arguments...]
```

Examples:

```bash
./commands/run.command doctor
./commands/run.command self-test
./commands/run.command prepare-voicebank
./commands/run.command probe-yv-final-chain
./commands/run.command train-ai-control-foundation
```

List every available command:

```bash
./commands/run.command list
```

Search by name:

```bash
./commands/run.command find yv
./commands/run.command find highband
```

Run the launcher without arguments to see commands grouped by purpose.

## Compatibility aliases

The launcher keeps a small number of old names that were previously root-level wrappers:

- `deep-train-ai-voicebank` -> `deep-train-voicebank`
- `highband-nyquist-diagnostic` -> `highband-test`

Direct execution of `scripts/*.command` is still possible for development and debugging, but documentation and normal use should prefer `commands/run.command`.
