#!/bin/bash
# Build the HIP library for the AMD GPU in this machine, then run the kernel tests and the kernel
# benchmark: one command for a rented AMD host.
#
#   PY=/path/to/python-with-rocm-torch engine/hip/amd/run_amd.sh [ROCM_ROOT]
#
# Run it from a checkout of this repository. The GPU architecture is read from rocminfo; set
# LX_HIP_ARCH (gfx1100, gfx1201, gfx942, ...) to override. Results are written to out/amd/.
set -euo pipefail
here=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$here"
rocm=${1:-${ROCM_PATH:-/opt/rocm}}
PY=${PY:?set PY to a Python whose torch is built for ROCm}
arch=${LX_HIP_ARCH:-$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | grep -v '^gfx000$' | head -1 || true)}
if [ -z "$arch" ]; then echo "no AMD GPU found by rocminfo; set LX_HIP_ARCH" >&2; exit 2; fi
clang="$rocm/lib/llvm/bin/amdclang++"
[ -x "$clang" ] || clang="$rocm/llvm/bin/amdclang++"
if [ ! -x "$clang" ]; then echo "no amdclang++ under $rocm" >&2; exit 2; fi

out=out/amd
bdir=build/amd-$arch
mkdir -p "$out"
echo "arch $arch, compiler $clang" | tee "$out/summary.txt"
cmake -S engine/hip -B "$bdir" -DLX_HIP_PLATFORM=amd -DCMAKE_CXX_COMPILER="$clang" \
  -DLX_HIP_ROCM_PATH="$rocm" -DLX_HIPBLASLT_INCLUDE_HINT="$rocm/include" -DLX_HIP_ARCHS="$arch" \
  > "$out/configure.log" 2>&1
cmake --build "$bdir" -j "$(nproc 2>/dev/null || echo 4)" > "$out/build.log" 2>&1
lib="$here/$bdir/libdromos_hip.so"
echo "built $lib" | tee -a "$out/summary.txt"

set +e
LX_KERNELS="$lib" "$PY" -m pytest -q tests/kernels > "$out/tests.log" 2>&1
tests=$?
"$PY" engine/hip/bench/bench_kernels.py "$lib" > "$out/bench.txt" 2>&1
bench=$?
set -e
tail -1 "$out/tests.log" | tee -a "$out/summary.txt"
echo "tests exit $tests, bench exit $bench (results in $out/)" | tee -a "$out/summary.txt"
exit $(( tests != 0 || bench != 0 ))
