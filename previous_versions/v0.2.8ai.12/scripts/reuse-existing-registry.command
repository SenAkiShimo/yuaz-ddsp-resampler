#!/bin/bash
set -euo pipefail
cat <<'TXT'
RC3.3 no longer requires manual registry reuse.
The voicebank-local ACTIVE generation is the source of truth; the global registry is only a rebuildable accelerator.
Nothing was changed.
TXT
