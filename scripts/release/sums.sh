#!/usr/bin/env bash
# Writes SHA256SUMS (sha256sum format) for the release assets in a directory (docs/RELEASE.md): every
# dromos-*.tar.gz and dromos-*.zip, and install.sh and install.ps1 when they are there.
#
#   scripts/release/sums.sh DIR
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: scripts/release/sums.sh DIR" >&2
  exit 2
fi
cd "$1"
assets=()
for f in dromos-*.tar.gz dromos-*.zip install.sh install.ps1; do
  [ -f "$f" ] && assets+=("$f")
done
if [ ${#assets[@]} -eq 0 ]; then
  echo "sums.sh: no release assets in $1" >&2
  exit 1
fi
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${assets[@]}" > SHA256SUMS
else
  shasum -a 256 "${assets[@]}" > SHA256SUMS
fi
cat SHA256SUMS
