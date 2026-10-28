#!/usr/bin/env bash
# Builds one release flavor into a tarball (docs/RELEASE.md): lernex-FLAVOR.tar.gz, which unpacks with no top
# directory to bin/lernex, lib/ (the backends, and the CUDA runtime for the CUDA flavors), share/lernex/tuning,
# share/lernex/channel (tarball) and LICENSE when the repository has one.
#
#   scripts/release/build.sh FLAVOR OUTDIR
#
# FLAVOR is linux-x86_64-cuda12, linux-x86_64-cuda13, linux-x86_64-cpu, linux-aarch64-cpu or darwin-arm64. Run it in the
# build base for the flavor (docs/RELEASE.md, "Build bases"): manylinux_2_28 for the Linux flavors, the CUDA toolkit of
# the flavor under CUDA_HOME (default /usr/local/cuda), and macOS 14 SDK tools on the Mac.
#
# Environment:
#   LX_VERSION      the version to report, X.Y.Z (default: the CMake rule, git describe --tags --match 'v*')
#   LX_GIT_SHA      the commit's short id, when the checkout has no git (default: git rev-parse)
#   LX_JOBS         build parallelism (default 2: the GPU server also hosts CI runners)
#   LX_ALLOW_DIRTY  1 to build an uncommitted tree (refused by default, so a release is a commit)
#   CUDA_HOME       the CUDA toolkit for the cuda flavors (default /usr/local/cuda)
#   SOURCE_DATE_EPOCH  the tarball's timestamps (default: the commit time, or now)
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: scripts/release/build.sh FLAVOR OUTDIR" >&2
  exit 2
fi
flavor=$1
outdir=$2
src=$(cd "$(dirname "$0")/../.." && pwd)
jobs=${LX_JOBS:-2}
cuda=${CUDA_HOME:-/usr/local/cuda}

# The arguments and targets each flavor is built with.
# No C++20 module scanning: the engine has no modules, and the scan is a compile per source file.
cmake_args=(-DLX_FLAVOR="$flavor" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_SCAN_FOR_MODULES=OFF)
targets=(lernex)
case "$flavor" in
  linux-x86_64-cuda12 | linux-x86_64-cuda13)
    cmake_args+=(-DLX_BACKEND_CUDA=ON -DLX_BACKEND_CPU=ON -DLX_BUNDLE_CUDA=ON -DLX_ARCHS=all -DLX_STATIC_LIBSTDCXX=ON
                 -DCUDAToolkit_ROOT="$cuda" -DCMAKE_CUDA_COMPILER="$cuda/bin/nvcc")
    targets+=(lernex_kernels lernex_cpu)
    ;;
  linux-x86_64-cpu | linux-aarch64-cpu)
    cmake_args+=(-DLX_BACKEND_CUDA=OFF -DLX_BACKEND_CPU=ON -DLX_STATIC_LIBSTDCXX=ON)
    targets+=(lernex_cpu)
    ;;
  darwin-arm64)
    cmake_args+=(-DLX_BACKEND_CUDA=OFF -DLX_BACKEND_CPU=OFF -DLX_BACKEND_METAL=ON -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0)
    targets+=(lernex_metal)
    ;;
  *)
    echo "build.sh: unknown flavor $flavor" >&2
    exit 2
    ;;
esac

if [ -n "${LX_VERSION:-}" ]; then cmake_args+=(-DLX_VERSION="$LX_VERSION"); fi
if [ -n "${LX_GIT_SHA:-}" ]; then export LX_GIT_SHA; fi

# A release is a commit: refuse a checkout with uncommitted changes, which would not be the tag.
if git -C "$src" rev-parse --git-dir >/dev/null 2>&1; then
  if [ -n "$(git -C "$src" status --porcelain)" ] && [ "${LX_ALLOW_DIRTY:-0}" != 1 ]; then
    echo "build.sh: the checkout has uncommitted changes; commit them or set LX_ALLOW_DIRTY=1" >&2
    exit 1
  fi
fi

mkdir -p "$outdir"
outdir=$(cd "$outdir" && pwd)
build="$outdir/build-$flavor"
stage="$outdir/stage-$flavor"
rm -rf "$build" "$stage"

# Clean configure and build of the flavor's own targets, then the install tree.
cmake -S "$src" -B "$build" -G Ninja "${cmake_args[@]}"
nice -n 5 cmake --build "$build" --target "${targets[@]}" -j "$jobs"
cmake --install "$build" --prefix "$stage" >/dev/null

# The packager says how this install was made (docs/RELEASE.md: share/lernex/channel).
mkdir -p "$stage/share/lernex"
printf 'tarball\n' > "$stage/share/lernex/channel"

# Our own binaries lose their local symbols; the exported lx_* symbols and the dynamic tables stay. The toolkit's
# libraries are NVIDIA's, shipped as they are.
if [ "$(uname -s)" = Darwin ]; then
  find "$stage/bin" "$stage/lib" -type f \( -name lernex -o -name 'liblernex_*' \) -exec strip -x {} +
else
  find "$stage/bin" "$stage/lib" -type f \( -name lernex -o -name 'liblernex_*' \) -exec strip --strip-unneeded {} +
fi

# The tree must be whole before it is packed.
for required in bin/lernex share/lernex/channel; do
  [ -e "$stage/$required" ] || { echo "build.sh: $required is missing from the install tree" >&2; exit 1; }
done
if ! ls "$stage"/share/lernex/tuning/*.json >/dev/null 2>&1; then
  echo "build.sh: no tuning tables in the install tree" >&2
  exit 1
fi
if [ ! -f "$stage/LICENSE" ]; then
  echo "build.sh: warning: the repository has no LICENSE, so the tarball has none" >&2
fi

# Packed with no top directory and fixed owners, so the same tree gives the same bytes.
tarball="$outdir/lernex-$flavor.tar.gz"
rm -f "$tarball"
top=()
while IFS= read -r entry; do top+=("$entry"); done < <(cd "$stage" && ls -A)
mtime=${SOURCE_DATE_EPOCH:-$(git -C "$src" log -1 --format=%ct 2>/dev/null || date +%s)}
if tar --version 2>/dev/null | grep -q GNU; then
  (cd "$stage" && tar --owner=0 --group=0 --numeric-owner --sort=name --mtime="@$mtime" -cf - "${top[@]}") | gzip -9n > "$tarball"
else
  (cd "$stage" && tar --uid 0 --gid 0 --uname "" --gname "" -cf - "${top[@]}") | gzip -9n > "$tarball"
fi
echo "build.sh: $tarball $(du -h "$tarball" | cut -f1) ($(ls "$stage/lib" | wc -l | tr -d ' ') libraries in lib/)"
