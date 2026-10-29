#!/bin/bash
# Build and test on the GPU server from this checkout.
#
#   scripts/remote.sh SLOT build              configure + build in the slot (Release, sm_120)
#   scripts/remote.sh SLOT test [PYTEST ARGS] build, then the kernel tests (under the GPU lock) and the C++ unit tests
#   scripts/remote.sh SLOT run 'CMD'          run a shell command in the slot after syncing ($PY is the venv python)
#   scripts/remote.sh SLOT x 'CMD'            the same, holding the GPU lock: use it for anything that loads a
#                                             whole model or measures speed, so runs never share the card
#
# LX_ARCHS=all (or any CMake list, see CMakeLists.txt) builds every target the engine runs on; the build
# then goes to build-LX_ARCHS-SAFE next to the default build/, so both can coexist in a slot.
# LX_BUILD_DIR names the build directory outright. LX_JOBS sets the build's parallelism (default 2: the
# server also hosts lernex-app's CI runners, and an out-of-memory compile has killed them before).
#
# The GPU lock only orders runs that take it. Inside it, x and test also wait while any other process
# holds the card (a vLLM sweep run by hand, say), unless LX_GPU_WAIT=0.
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
# Changed files are found by checksum and land with the server's current time, not the Mac's: a copied mtime can
# be older than the object built from the previous version, and ninja then links the stale object.
rsync -rlpzc --no-times --delete --exclude 'build*/' --exclude out/ --exclude .git/ --exclude __pycache__/ --exclude .pytest_cache/ "$here/" "$host:$remote/"
bdir=${LX_BUILD_DIR:-build}
cfg=""
if [ -n "${LX_ARCHS:-}" ]; then
  cfg="-DLX_ARCHS=$LX_ARCHS"
  [ -n "${LX_BUILD_DIR:-}" ] || bdir="build-$(printf '%s' "$LX_ARCHS" | tr -c 'A-Za-z0-9\n' '-')"
fi
env="export PATH=/usr/local/cuda/bin:\$PATH; export LX_KERNELS=\$PWD/$bdir/libdromos_kernels.so; export PY=\$HOME/venvs/hfport/bin/python"
# Every slot compiles inside the server's lernex-build.slice where it is installed (scripts/server/lernex-build.slice:
# MemoryHigh 14G, MemoryMax 16G across all slots), so an out-of-memory compile kills a compile, never the Windows VM
# or a GPU run. The slots also build one at a time, under the build lock: four slots compiling the decode kernels'
# family copies together (up to 6 GB of ptxas each) sat at the slice's cap with swap full for 50 minutes and finished
# nothing, where one at a time takes about 10.
slice="\$(systemctl --user cat lernex-build.slice >/dev/null 2>&1 && echo systemd-run --user --scope -q --slice=lernex-build.slice)"
build="cmake -S . -B $bdir -G Ninja -DCMAKE_BUILD_TYPE=Release $cfg >/dev/null && { flock -n \$HOME/lernex-engine-dev/.build.lock true || echo '[remote] waiting for the build lock: another slot is compiling' >&2; } && flock \$HOME/lernex-engine-dev/.build.lock nice -n 5 $slice cmake --build $bdir -j ${LX_JOBS:-2}"
# Inside the lock: wait until no other process holds the card.
gpuwait="[ ${LX_GPU_WAIT:-1} = 0 ] || "'{ n=0; while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do [ $n = 0 ] && echo "[remote] waiting for the GPU: another process holds it" >&2; n=1; sleep 30; done; }'
q() { [ $# -eq 0 ] || printf '%q ' "$@"; }  # nothing for no arguments: an empty '' would be a path
case "$verb" in
  build) ssh "$host" "cd $remote && $env && $build" ;;
  # The kernel tests allocate gigabytes on the card, so they hold the GPU lock as model runs do.
  test) ssh "$host" "cd $remote && $env && $build && flock \$HOME/lernex-engine-dev/.gpu.lock bash -c $(q "$gpuwait; \$PY -m pytest -q tests/kernels $(q "$@")") && { [ ! -x $bdir/dromos_unit ] || $bdir/dromos_unit; }" ;;
  run) ssh "$host" "cd $remote && $env && $*" ;;
  x) ssh "$host" "cd $remote && $env && flock \$HOME/lernex-engine-dev/.gpu.lock bash -c $(q "$gpuwait; $*")" ;;
  *) echo "unknown verb $verb" >&2; exit 2 ;;
esac
