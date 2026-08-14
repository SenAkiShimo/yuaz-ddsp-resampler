# Contributing

Contributions are welcome through issues and pull requests.

## Development setup

```bash
./setup-macos.command
./configure-macos.command
```

## Required checks

Before opening a pull request, run:

```bash
python3 -m compileall -q src/yuaz_ddsp_resampler
./self-test.command

for f in *.command scripts/*.command; do
  bash -n "$f"
done
```

Changes to synthesis, crossover, high-band processing, state migration, or OpenUtau routing should include a regression test when practical.

## Repository hygiene

Do not commit:

- `.venv/`
- `config.json`
- logs, lock files, PID/state caches, or generated registries
- voicebank `.yuaz-*` state
- WAV test renders
- trained `.pt`, `.pth`, or `.ckpt` files unless their redistribution rights are explicitly documented
- datasets or dataset-derived weights without confirming their licenses

Keep public comments focused on implementation behavior, invariants, and non-obvious constraints.
