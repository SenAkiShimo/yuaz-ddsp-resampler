#!/bin/bash
set -euo pipefail
cat <<'TXT'
RC3.3 does not support deleting active acoustic state in place.
Nothing was deleted.
Use prepare-voicebank.command to create and atomically switch to a replacement generation.
External backups and previous generations are kept deliberately.
TXT
