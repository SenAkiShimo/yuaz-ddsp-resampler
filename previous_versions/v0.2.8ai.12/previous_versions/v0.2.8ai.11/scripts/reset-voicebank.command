#!/bin/bash
set -euo pipefail
cat <<'TXT'
RC3.3 intentionally disables destructive in-place voicebank reset.
Nothing was deleted.

To replace a bad trained state safely, run prepare-voicebank.command and choose:
  1) Adopt RC3.2 Baseline, or
  2) CLEAN DEEP + Stage C Fidelity.

RC3.3 builds a new generation, validates it, and switches ACTIVE atomically.
Old RC3.2 state and previous RC3.3 generations are preserved for rollback.
TXT
