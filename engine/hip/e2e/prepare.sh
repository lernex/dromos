#!/bin/bash
# Make the ported model tree that engine/hip/e2e builds: the repository's engine/core, engine/model and
# cli/ with docs/handoff/hip-metis.diff applied. The repository itself is not changed.
#
#   engine/hip/e2e/prepare.sh OUT_DIR
set -euo pipefail
here=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$here"
out=${1:?usage: engine/hip/e2e/prepare.sh OUT_DIR}
rm -rf "$out"
mkdir -p "$out/engine" "$out/cli"
cp -R engine/core engine/model "$out/engine/"
cp cli/*.cpp "$out/cli/"
patch -s -p1 -d "$out" < docs/handoff/hip-metis.diff
echo "ported tree at $out"
