# Releases and installs

How a tagged commit becomes something people install, and the contract every installer shares.
Every install path ends in the same tree, so `lernex` behaves the same however it arrived.

## Versions

A release is a git tag `vX.Y.Z` on `main` (`vX.Y.Z-rcN` for a pre-release). The build bakes the
version in: CMake takes `-DLX_VERSION=X.Y.Z`, and without it reads `git describe --tags --match 'v*'`
(an untagged build is `0.0.0-dev+<sha>`). `lernex --version` prints one line and loads no backend:

    lernex 0.1.0 (linux-x86_64-cuda12, 777d965)

## Release assets

Names carry no version, so `https://github.com/lernex/lernex-engine/releases/latest/download/NAME`
is a stable link:

| Asset | Contents |
|---|---|
| `lernex-linux-x86_64-cuda12.tar.gz` | CUDA backend, every architecture (`-DLX_ARCHS=all`), plus the CPU backend |
| `lernex-linux-x86_64-cpu.tar.gz` | CPU backend only |
| `lernex-linux-aarch64-cpu.tar.gz` | CPU backend only |
| `lernex-darwin-arm64.tar.gz` | Metal backend (macOS 14+) |
| `install.sh` | the installer |
| `SHA256SUMS` | `sha256sum` format, one line per asset above |

One CUDA flavor ships, built with CUDA 12.8. Measured 2026-10-28 on an RTX 5070 Ti (driver 595), Praxis:
the 12.8 build decodes 293.9 / 1,140 tok/s at 1 / 8 streams against 295.2 / 1,165 for a 13.4 build, with
the same KL to the reference, and 12.8 runs on older drivers. The driver it needs: 570.26 or newer for the
12.8 toolkit (and for any RTX 50 card), 525.60.13 or newer for older cards through NVIDIA's minor-version
compatibility (the build carries real code for every architecture, no PTX to JIT; drivers below 595 are
untested). An older driver, or no GPU, gets the CPU flavor. `scripts/release/build.sh` still builds a
`cuda13` flavor, and install.sh prefers it when a release lists one.

## The tree

Every tarball unpacks, with no top directory, to:

    bin/lernex
    lib/liblernex_kernels.so  liblernex_cpu.so      (linux; liblernex_metal.dylib on macOS)
    lib/libcudart.so.N  lib/libcublasLt.so.N      (CUDA flavor only, the toolkit's own)
    share/lernex/tuning/*.json
    share/lernex/channel                           tarball | homebrew | pip | docker
    LICENSE

`bin/lernex` finds its backends in `../lib` (already done by engine/dispatch), and the backends find
the bundled CUDA libraries through an `$ORIGIN` rpath. Tuning tables are read from `../share/lernex/tuning`
next to the real executable, then the source tree they were built from. Nothing reads a path baked in at
build time unless it exists.

`share/lernex/channel` says who installed it, so `lernex update` and the daily notice name the right
command: `lernex update` for `tarball`, `brew upgrade lernex`, `pip install -U lernex`,
`docker pull ghcr.io/lernex/lernex`. A tree without the file is a source build.

## Where things live

| Path | What |
|---|---|
| `~/.lernex/versions/X.Y.Z/` | an installed tree (install.sh and `lernex update`) |
| `~/.lernex/current` | symlink to the active version; switched atomically |
| `~/.local/bin/lernex` | symlink to `~/.lernex/current/bin/lernex` (install.sh adds it to PATH if needed) |
| `~/.cache/lernex/models/` | models (`LERNEX_MODELS` overrides) |
| `~/.cache/lernex/update-check` | the daily check's cache |

Environment: `LERNEX_NO_UPDATE_CHECK=1` turns the daily check off. `LERNEX_RELEASES` (default
`https://github.com/lernex/lernex-engine/releases`) points install.sh and `lernex update` at another
release host, for tests.

## Updates

- `lernex update` finds the latest release, downloads this install's flavor, checks it against
  `SHA256SUMS`, unpacks it to `~/.lernex/versions/NEW`, flips `current`, and keeps the previous
  version for one rollback. `lernex update --check` only reports.
- `run` and `serve` print one quiet line when a newer release is known: at most one network check a
  day, in the background with a short timeout, cached, so startup never waits on it.

## Install paths

    curl -fsSL https://github.com/lernex/lernex-engine/releases/latest/download/install.sh | sh
    brew install lernex/tap/lernex
    pip install lernex
    docker run --gpus all -p 8080:8080 -v lernex:/root/.cache/lernex ghcr.io/lernex/lernex praxis

install.sh picks the flavor from the OS, the architecture and the NVIDIA driver; re-running it upgrades.

## Build bases

Linux builds run on an old base so the binaries run on any current distribution: manylinux_2_28
(glibc 2.28) where the CUDA toolkit allows it, else ubuntu 22.04. The CPU backend compiles every ISA
level and picks at run time (engine/cpu); nothing is built with `-march=native`. macOS builds target 14.0.
TLS comes from the system libcurl, opened at run time (engine/core/http.cpp), so no OpenSSL is linked.
