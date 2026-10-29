#!/usr/bin/env bash
# Installs the CUDA toolkit a CUDA release flavor is built with, into the manylinux_2_28 build base (docs/RELEASE.md),
# from NVIDIA's redistributable archives. Only what the engine compiles and links: the compiler, the runtime and its
# headers, cuBLAS (cuBLASLt), CCCL (CUB), and the documentation archive for EULA.txt, the license the bundled cudart
# and cuBLASLt ship under. The toolkit lands in /usr/local/cuda-MAJOR.MINOR (CUDA_HOME for build.sh).
#
#   scripts/release/cuda-toolkit.sh 12.8    # linux-x86_64-cuda12
#   scripts/release/cuda-toolkit.sh 13.4    # linux-x86_64-cuda13
#
# Each series is pinned to one of NVIDIA's redistrib manifests, and the manifest to its sha256, so a release builds
# against exactly the versions its candidate was tested with, and every archive is checked against the manifest's
# sha256. NVIDIA's package repositories are not used: their index is republished in place (on 2026-10-29 it named
# files that were not there for hours), and they serve the newest packages of a series, which can change between a
# release candidate and its release.
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: scripts/release/cuda-toolkit.sh MAJOR.MINOR" >&2
  exit 2
fi
version=$1
redist=https://developer.download.nvidia.com/compute/cuda/redist
platform=linux-x86_64

# Series, manifest, the manifest's sha256, and the archives. 12.8.2 holds nvcc 12.8.93, cudart and CCCL 12.8.90 and
# cuBLAS 12.8.5.5, the versions the 12.8 packages installed for v0.1.0-rc1 and the cuBLAS the pip wheel requires
# (packaging/pip/build_wheel.py, CUDA_WHEELS). CUDA 13 splits crt and NVVM out of nvcc and renames CCCL.
case "$version" in
  12.8)
    manifest=redistrib_12.8.2.json
    manifest_sha256=bbf0798e9aabdd6887b429f0d1ff27412bca2ead081f898a18f0381cd8cb7359
    archives=(cuda_nvcc cuda_cudart cuda_cccl libcublas cuda_documentation)
    ;;
  13.4)
    manifest=redistrib_13.4.3.json
    manifest_sha256=1420d5c676cad98a5ff423d529f116fee92f6b99559a3afdad41e782d6a11dba
    archives=(cuda_nvcc cuda_crt libnvvm cuda_cudart cccl libcublas cuda_documentation)
    ;;
  *)
    echo "cuda-toolkit.sh: no pinned manifest for CUDA $version (add one above)" >&2
    exit 2
    ;;
esac

prefix=/usr/local/cuda-$version
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
# A loop rather than curl's --retry-all-errors, which the base's curl 7.61 does not have.
fetch() {
  local try
  for try in 1 2 3 4 5 6; do
    curl -fsSL -o "$2" "$1" && return 0
    echo "cuda-toolkit.sh: fetching $1 failed (try $try of 6)" >&2
    sleep $((try * 10))
  done
  return 1
}

fetch "$redist/$manifest" "$work/manifest.json"
echo "$manifest_sha256  $work/manifest.json" | sha256sum -c --quiet - || {
  echo "cuda-toolkit.sh: $manifest does not match its pinned sha256" >&2; exit 1; }

# One line per archive: its path under redist/ and its sha256, as the manifest gives them.
python3 - "$work/manifest.json" "$platform" "${archives[@]}" > "$work/archives.txt" <<'EOF'
import json, sys
manifest, platform, names = sys.argv[1], sys.argv[2], sys.argv[3:]
data = json.load(open(manifest))
for name in names:
    entry = data.get(name, {}).get(platform)
    if not entry:
        sys.exit("cuda-toolkit.sh: the manifest has no %s for %s" % (name, platform))
    print(entry["relative_path"], entry["sha256"], data[name]["version"])
EOF

mkdir -p "$prefix"
while read -r path sha256 component_version; do
  file=$work/$(basename "$path")
  fetch "$redist/$path" "$file"
  echo "$sha256  $file" | sha256sum -c --quiet - || { echo "cuda-toolkit.sh: $path does not match its sha256" >&2; exit 1; }
  # Each archive holds one top folder (name-platform-version-archive/) with bin/, include/, lib/ and the rest.
  tar -xJf "$file" -C "$prefix" --strip-components=1
  rm -f "$file"
  echo "cuda-toolkit.sh: $(basename "$path" .tar.xz) ($component_version)"
done < "$work/archives.txt"

# The archives put libraries in lib/; the packaged toolkit and its tools look in lib64/ too.
[ -e "$prefix/lib64" ] || ln -s lib "$prefix/lib64"
[ -x "$prefix/bin/nvcc" ] || { echo "cuda-toolkit.sh: no nvcc under $prefix" >&2; exit 1; }
[ -f "$prefix/EULA.txt" ] || { echo "cuda-toolkit.sh: no EULA.txt under $prefix" >&2; exit 1; }
"$prefix/bin/nvcc" --version | tail -1
