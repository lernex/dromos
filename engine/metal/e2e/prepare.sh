#!/bin/bash
# Make the ported model tree that engine/metal/e2e builds: the repository's engine/core, engine/model and
# cli/ with docs/handoff/metal-metis.diff applied. The repository itself is not changed.
#
#   engine/metal/e2e/prepare.sh OUT_DIR
set -euo pipefail
here=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$here"
out=${1:?usage: engine/metal/e2e/prepare.sh OUT_DIR}
rm -rf "$out"
mkdir -p "$out/engine" "$out/cli"
cp -R engine/core engine/model "$out/engine/"
cp cli/*.cpp "$out/cli/"
patch -s -E -p1 -d "$out" < docs/handoff/metal-metis.diff
echo "ported tree at $out"
