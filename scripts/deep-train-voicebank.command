#!/bin/bash
set -euo pipefail
cat <<'TXT'
Yuaz 0.2.8ai.15 does not write voicebank training state.

This control-calibration build reads the existing .yuaz-0.2.8ai14 generation and
its .ai14 trained artifacts without modifying them. Deep training is disabled so
the 0.2.8ai.14 baseline remains byte-for-byte recoverable for A/B comparison.

Run Deep from 0.2.8ai.14 if you need to retrain the current voicebank baseline.
TXT
exit 2
