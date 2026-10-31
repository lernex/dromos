# Releases and installs

How a tagged commit becomes something people install, and the contract every installer shares.
Every install path ends in the same tree, so `dromos` behaves the same however it arrived.

## Versions

A release is a git tag `vX.Y.Z` on `main` (`vX.Y.Z-rcN` for a pre-release). The build bakes the
version in: CMake takes `-DLX_VERSION=X.Y.Z`, and without it reads `git describe --tags --match 'v*'`
(an untagged build is `0.0.0-dev+<sha>`). `dromos --version` prints one line and loads no backend:

    dromos 0.1.0 (linux-x86_64-cuda12, 777d965)

## Release assets

Names carry no version, so `https://github.com/lernex/dromos/releases/latest/download/NAME`
is a stable link:

| Asset | Contents |
|---|---|
| `dromos-linux-x86_64-cuda13.tar.gz` | CUDA 13.4 backend, every architecture (`-DLX_ARCHS=all`), plus the CPU backend |
| `dromos-linux-x86_64-cuda12.tar.gz` | CUDA 12.8 backend, every architecture, plus the CPU backend |
| `dromos-linux-x86_64-cpu.tar.gz` | CPU backend only |
| `dromos-linux-aarch64-cpu.tar.gz` | CPU backend only |
| `dromos-darwin-arm64.tar.gz` | Metal backend (macOS 14+) |
| `dromos-windows-x86_64-cuda13.zip` | CUDA 13.4 backend, every architecture, plus the CPU backend (Windows 10 1809+ x64) |
| `dromos-windows-x86_64-cuda12.zip` | CUDA 12.8 backend, every architecture, plus the CPU backend (Windows 10 1809+ x64) |
| `dromos-windows-x86_64-cpu.zip` | CPU backend only |
| `install.sh` | the installer (Linux, macOS) |
| `install.ps1` | the installer (Windows) |
| `SHA256SUMS` | `sha256sum` format, one line per asset above |

Linux ships two CUDA flavors. `cuda13` is built with CUDA 13.4 (cuBLAS 13.8), the toolkit the engine is
developed and measured on, and install.sh gives it to any driver of 580 or newer. Measured 2026-10-30 on an RTX
5070 Ti (driver 595), the same commit built both ways, the 12.8 build decodes 3 to 5% slower at one stream
and 10 to 17% slower at 32 streams across Praxis BF16, INT8, FP8 and NVFP4 and Logos FP8. `cuda12` is built
with CUDA 12.8 for older drivers: 570.26 or newer for the 12.8 toolkit (and for any RTX 50 card), 525.60.13 or
newer for older cards through NVIDIA's minor-version compatibility (the builds carry real code for every
architecture, no PTX to JIT; drivers below 595 are untested). Windows ships the same two, and install.ps1
chooses the same way: `cuda13` for a driver of 580 or newer, `cuda12` with floors of 570.65 for the 12.8
toolkit and 527.41 through minor-version compatibility. An older driver, or no GPU, gets the CPU flavor.

## The tree

Every archive unpacks, with no top directory, to:

    bin/dromos
    lib/libdromos_kernels.so  libdromos_cpu.so      (linux; libdromos_metal.dylib on macOS)
    lib/libcudart.so.N  lib/libcublasLt.so.N      (CUDA flavor only, the toolkit's own)
    share/dromos/tuning/*.json
    share/dromos/channel                           tarball | homebrew | pip | docker
    share/dromos/licenses/                         third-party licenses (and NVIDIA's EULA, CUDA flavor)
    LICENSE  NOTICE                                Apache-2.0, and what else ships under which terms

The Windows zip is the same tree with the backends beside the engine, where Windows looks for a program's DLLs:

    bin\dromos.exe  dromos_cpu.dll  dromos_kernels.dll
    bin\cudart64_12.dll  cublasLt64_12.dll          (CUDA flavor only, the toolkit's own)
    share\dromos\...  LICENSE  NOTICE              as above

`bin/dromos` finds its backends in `../lib` (already done by engine/dispatch), and the backends find the bundled
CUDA libraries through an `$ORIGIN` rpath. On Windows the engine loads each backend with its own folder searched
first for what it links (`LoadLibraryExW`, engine/platform/posix_win.cpp), so the CUDA backend finds the DLLs beside
it, then `PATH` (where pip's NVIDIA wheels are put). Tuning tables are read from `../share/dromos/tuning` next to
the real executable, then the source tree they were built from. Nothing reads a path baked in at build time unless
it exists.

`share/dromos/channel` says who installed it, so `dromos update` and the daily notice name the right
command: `dromos update` for `tarball`, `brew upgrade dromos`, `pip install -U dromos`,
`docker pull ghcr.io/lernex/dromos`. A tree without the file is a source build.

## Where things live

| Path | What |
|---|---|
| `~/.dromos/versions/X.Y.Z/` | an installed tree (install.sh and `dromos update`) |
| `~/.dromos/current` | symlink to the active version; switched atomically |
| `~/.local/bin/dromos` | symlink to `~/.dromos/current/bin/dromos` (install.sh adds it to PATH if needed) |
| `~/.cache/dromos/models/` | models (`DROMOS_MODELS` overrides) |
| `~/.cache/dromos/update-check` | the daily check's cache |

On Windows (engine/platform/platform.cpp and install.ps1 name these folders, and only they do):

| Path | What |
|---|---|
| `%LOCALAPPDATA%\Programs\Dromos\versions\X.Y.Z\` | an installed tree (install.ps1 and `dromos update`) |
| `%LOCALAPPDATA%\Programs\Dromos\current`, `previous` | directory junctions; `current` is retargeted in place |
| the user `PATH` | gains `%LOCALAPPDATA%\Programs\Dromos\current\bin` once (install.ps1) |
| `%LOCALAPPDATA%\Dromos\models\` | models (`DROMOS_MODELS` overrides) |
| `%LOCALAPPDATA%\Dromos\update-check`, `tune\` | the daily check's cache, tuning results |

Programs go under `Programs`, as per-user installs do on Windows; data that can be fetched again goes beside it.
`~` in a path given on the command line is `%USERPROFILE%`.

Environment: `DROMOS_NO_UPDATE_CHECK=1` turns the daily check off. `DROMOS_RELEASES` (default
`https://github.com/lernex/dromos/releases`) points install.sh and `dromos update` at another
release host, for tests.

## Updates

- `dromos update` finds the latest release, downloads this install's flavor, checks it against
  `SHA256SUMS`, unpacks it to `~/.dromos/versions/NEW`, flips `current`, and keeps the previous
  version for one rollback. `dromos update --check` only reports.
- On Windows `current` is a junction, retargeted in place. A version that is still running (a `dromos serve` from
  before the update) is renamed out of `versions` (Windows lets the folder of a running program move) and deleted
  once it stops: the next update or install sweeps what it could not remove.
- `run` and `serve` print one quiet line when a newer release is known: at most one network check a
  day, in the background with a short timeout, cached, so startup never waits on it.

## Install paths

    curl -fsSL https://github.com/lernex/dromos/releases/latest/download/install.sh | sh
    irm https://github.com/lernex/dromos/releases/latest/download/install.ps1 | iex      (Windows PowerShell)
    brew trust --formula lernex/tap/dromos && brew install lernex/tap/dromos
    pip install dromos
    docker run --gpus all -p 8080:8080 -v dromos:/root/.cache/dromos ghcr.io/lernex/dromos praxis

install.sh picks the flavor from the OS, the architecture and the NVIDIA driver; re-running it upgrades.
install.ps1 does the same on Windows from `nvidia-smi` (driver and compute capability), needs no administrator
rights, and puts `current\bin` on the user `PATH`. Both honour `DROMOS_VERSION`, `DROMOS_RELEASES`, and
`DROMOS_FLAVOR` (`cpu`, `cuda12` or `cuda13`).
Homebrew 7 loads a formula from a third-party tap only once it is trusted, so the brew install is two
commands: `brew trust --formula lernex/tap/dromos` once, then `brew install lernex/tap/dromos`. The tap is
github.com/lernex/homebrew-tap; its `Formula/dromos.rb` is the one in `packaging/homebrew-tap`.

## Cutting a release

1. Tag `vX.Y.Z` on `main` and push the tag. `release.yml` builds every flavor and drafts the GitHub release
   with the tarballs and zips, `install.sh`, `install.ps1` and `SHA256SUMS`.
   The draft can also be built off GitHub, as v0.1.0 was: each flavor with `build.sh` or `build.ps1` in its base
   (manylinux_2_28 containers on the Lernex server, macOS on a Mac, Windows on the Windows VM), the installers
   copied beside them and `scripts/release/sums.sh` run over the folder, then
   `gh release create vX.Y.Z --draft --target <commit> --title "Dromos X.Y.Z" <folder>/*`. A draft has no tag
   yet. Publishing it creates the tag, which starts `release.yml`, and its first job sees the release already
   exists, so nothing is built twice.
2. Check the draft's assets (install each flavor on a clean machine), then publish it. Publishing starts
   `pypi.yml` (the wheels, through the `pypi` environment) and `docker.yml` (the GHCR image).
3. Bump the tap: in a checkout of lernex/homebrew-tap, run
   `<engine checkout>/packaging/homebrew-tap/bump-formula.sh X.Y.Z Formula/dromos.rb`, commit and push. Nothing
   automates this step: it needs push access to the tap, and a CI deploy key for it is the owner's decision.
   Copy the bumped formula back into `packaging/homebrew-tap/Formula/dromos.rb` here so the two stay the same.
4. Check each channel installs the new version: `install.sh`, `install.ps1`, `brew upgrade dromos`,
   `pip install -U dromos`, `docker pull ghcr.io/lernex/dromos`, and `dromos update` from the previous release.

## Build bases

Linux builds run on an old base so the binaries run on any current distribution: manylinux_2_28
(glibc 2.28) where the CUDA toolkit allows it, else ubuntu 22.04. The CPU backend compiles every ISA
level and picks at run time (engine/cpu); nothing is built with `-march=native`. macOS builds target 14.0.
The CUDA flavors' toolkit comes from NVIDIA's redistributable archives, pinned by one manifest per series
(`scripts/release/cuda-toolkit.sh`: `redistrib_12.8.2.json` for 12.8, `redistrib_13.4.3.json` for 13.4) and each archive checked against its sha256, so a
release builds against exactly the versions its release candidate was tested with. NVIDIA's package repositories are
not used: they serve the newest packages of a series, and their index has been republished broken.
TLS comes from the system libcurl, opened at run time (engine/core/http.cpp), so no OpenSSL is linked.

Windows builds run on `windows-2022` with Visual Studio 2022's MSVC (`scripts/release/build.ps1`), the C runtime
linked statically so no Visual C++ redistributable is needed, and a manifest that makes the process UTF-8 and
long-path aware. The CUDA flavors use nvcc 12.8 and 13.4 with `cl` as their host compiler, from the same pinned
manifests as Linux (`scripts/release/cuda-toolkit.ps1`; the toolkit installer does not run unattended). TLS and proxies come from
WinHTTP, the system's own HTTP stack, so the certificate store and proxy settings are Windows's and nothing ships
for them. The zip is packed with fixed timestamps (`scripts/release/pack_zip.py`), so a commit gives the same bytes.
