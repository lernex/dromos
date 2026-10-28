#!/usr/bin/env bash
# Installs the CUDA toolkit a CUDA release flavor is built with, into the manylinux_2_28 build base (docs/RELEASE.md),
# from NVIDIA's RHEL 8 repository. Only what the engine compiles and links: the compiler, the runtime and its headers,
# cuBLAS (cuBLASLt), CCCL (CUB), and the documentation package for EULA.txt, the license the bundled cudart and
# cuBLASLt ship under. The toolkit lands in /usr/local/cuda-MAJOR.MINOR (CUDA_HOME for build.sh).
#
#   scripts/release/cuda-toolkit.sh 12.8    # linux-x86_64-cuda12
#   scripts/release/cuda-toolkit.sh 13.4    # linux-x86_64-cuda13
#
# Needs root in the base (dnf). The packages are the newest in the repository for the series.
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: scripts/release/cuda-toolkit.sh MAJOR.MINOR" >&2
  exit 2
fi
version=$1
series=${version//./-}
repo=https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64

curl -fsSL -o /etc/yum.repos.d/cuda-rhel8.repo "$repo/cuda-rhel8.repo"
rpm --import "$repo/D42D0685.pub"
dnf -y install --setopt=install_weak_deps=False \
  "cuda-nvcc-$series" "cuda-cudart-$series" "cuda-cudart-devel-$series" "cuda-driver-devel-$series" \
  "cuda-cccl-$series" "libcublas-$series" "libcublas-devel-$series" "cuda-documentation-$series"
dnf clean all >/dev/null
[ -x "/usr/local/cuda-$version/bin/nvcc" ] || { echo "cuda-toolkit.sh: no nvcc under /usr/local/cuda-$version" >&2; exit 1; }
[ -f "/usr/local/cuda-$version/EULA.txt" ] || { echo "cuda-toolkit.sh: no EULA.txt under /usr/local/cuda-$version" >&2; exit 1; }
"/usr/local/cuda-$version/bin/nvcc" --version | tail -1
