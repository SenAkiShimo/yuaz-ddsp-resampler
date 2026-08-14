#!/bin/bash
set -euo pipefail
echo "REFUSED: ai.14 does not purge predecessor versions."
echo "ai.13 must remain installed and its .yuaz-0.2.8ai13 states remain untouched."
exit 2
