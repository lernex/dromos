#!/bin/bash
# Build and test on the GPU server from this checkout.
#
#   scripts/remote.sh SLOT build              configure + build in the slot (Release, sm_120)
#   scripts/remote.sh SLOT test [PYTEST ARGS] build, then the kernel tests
#   scripts/remote.sh SLOT run 'CMD'          run a shell command in the slot after syncing ($PY is the venv python)
#   scripts/remote.sh SLOT x 'CMD'            the same, holding the GPU lock: use it for anything that loads a
#                                             whole model or measures speed, so runs never share the card
#
# Each SLOT is its own copy on the server (~/lernex-engine-dev/SLOT), so several people can
# build at once without stepping on each other. Use your own slot name.
set -euo pipefail
slot=${1:?slot}; shift
verb=${1:?verb}; shift
host=${LX_HOST:-lernex}
here=$(cd "$(dirname "$0")/.." && pwd)
remote="lernex-engine-dev/$slot"
ssh "$host" "mkdir -p $remote"
rsync -az --delete --exclude build/ --exclude out/ --exclude .git/ --exclude __pycache__/ --exclude .pytest_cache/ "$here/" "$host:$remote/"
env='export PATH=/usr/local/cuda/bin:$PATH; export LX_KERNELS=$PWD/build/liblernex_kernels.so; export PY=$HOME/venvs/hfport/bin/python'
build='cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null && nice -n 5 cmake --build build -j 6'
q() { printf '%q ' "$@"; }
case "$verb" in
  build) ssh "$host" "cd $remote && $env && $build" ;;
  test) ssh "$host" "cd $remote && $env && $build && \$PY -m pytest -q tests/kernels $(q "$@")" ;;
  run) ssh "$host" "cd $remote && $env && $*" ;;
  x) ssh "$host" "cd $remote && $env && flock \$HOME/lernex-engine-dev/.gpu.lock bash -c $(q "$*")" ;;
  *) echo "unknown verb $verb" >&2; exit 2 ;;
esac
