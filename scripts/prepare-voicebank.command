#!/bin/bash
set -euo pipefail
cat <<'TXT'
Yuaz 0.2.8ai.15 is a control-calibration runtime build.

Voicebank Prepare/Deep is intentionally disabled in this build because ai.15 reads
0.2.8ai.14 trained generations as a read-only compatibility source. This prevents
an ai.15 test from overwriting or mutating any .yuaz-0.2.8ai14 training state.

Use the installed 0.2.8ai.14 runtime if you need to Prepare/Continue/Deep a voicebank.
TXT
exit 2
