# Install tests

What `scripts/install.sh`, the Docker image and the install paths were run on, with the result of each run. Every row
names the machine, the release build it installed and what happened. A row marked "not run" has not been run in this
round; the Open items say what it needs.

## Method

- Release host: a directory laid out like GitHub's (`latest/download/` and `download/vX.Y.Z/`, each holding the
  tarballs, `SHA256SUMS` and `install.sh`), reached through `LERNEX_RELEASES`. `SHA256SUMS` is written by
  `scripts/release/sums.sh`, as the release workflow does.
- The tarballs are the release agent's `scripts/release/build.sh` outputs, not hand-assembled trees. The release ships
  one CUDA build, cuda12 (CUDA 12.8). The x86_64 CPU tarballs are the `out/v010` and `out/v011` builds on the GPU server;
  the cuda12 tarball is `out12/lernex-linux-x86_64-cuda12.tar.gz` (713,243,380 bytes, sha256 `572ee1be5e6aa531…`,
  `--version` `lernex 0.1.0 (linux-x86_64-cuda12, 0609dd3)`, built before the line-info cut). The local builds have no
  `LX_GIT_SHA` in the CPU runs (`wip`); the release workflow sets it from the commit.
- `install.sh` is the one on main at `9f13f51` (sha256 `e0fd2436…`).
- Each run starts from a fresh `HOME` or a fresh container, so the install is the first install on the machine.
- The Praxis weights come from the GPU server's cache. Containers mount that folder, so no 6 GB download happens in a
  test.
- Flags that were needed to get a run through, and why, are in the Notes.

## Results

| # | What | Where | Build | Result |
|---|------|-------|-------|--------|
| 1 | Fresh install 0.1.0, `--version`, `doctor`, `pull` without libcurl, one prompt, upgrade to 0.1.1 | ubuntu:22.04 x86_64, udocker on the GPU server, wget only | cpu, `build.sh` (out/v010, out/v011) | pass (1a) |
| 2 | Same as 1 | ubuntu:24.04 x86_64, udocker on the GPU server, curl | cpu, `build.sh` | pass (2a) |
| 3 | Generation, one prompt, thinking off | rows 1 and 2 | cpu | pass: "17 times 3" answered `51`, 30 and 32 tok/s |
| 4 | Darwin install, `--version`, `doctor`, Metal backend loads, temp HOME | this Mac, arm64 | metal, `build.sh` | not run in this round |
| 5 | Install refusals: Intel Mac, Rosetta shell, macOS 13 | this Mac, shimmed `uname`, `sysctl`, `sw_vers` | n/a | pass, each prints its reason and installs nothing (run before the link change; the refusals come before it) |
| 6 | Driver and GPU selection, cases A to L | GPU server, shimmed `nvidia-smi` and `uname` | fake tarballs | pass, rerun on the current `install.sh`, same outcomes as 6a |
| 7 | Upgrade, re-install with and without `.install`, previous link, piped install | this Mac, temp HOME | fake darwin tarballs | pass (7a) |
| 8 | Piped install, tty progress bar, idempotent PATH line, fish config | this Mac | darwin | pass (earlier run, before the link change) |
| 9 | Checksum mismatch, bad version, bad `LERNEX_FLAVOR`, a non-Lernex file at `~/.local/bin/lernex` | this Mac and GPU server | fake | pass: nothing installed in each, rerun on the current `install.sh` |
| 10 | GPU install through `install.sh` (cuda12 chosen by itself), `--version`, `doctor`, one prompt with `lernex run`, `serve` and one curl | GPU server, native, temp HOME, local release host | cuda12, `build.sh` (out12) | pass for install, run, serve and curl; `doctor praxis` fails in this build, see 10a |
| 11 | arm64 Ubuntu 22.04 and 24.04: install, `doctor`, upgrade | Docker Desktop, linux/arm64 | cpu, `build.sh` (aarch64) | not run in this round |
| 12 | amd64 Ubuntu 22.04 and 24.04 under emulation: install, `doctor`, upgrade | Docker Desktop, linux/amd64 | cpu, `build.sh` (x86_64) | not run in this round |
| 13 | Image build `linux/amd64` from `docker/Dockerfile` on main (cuda12 default), `--version` in the image, saved to the server | this Mac, Docker Desktop, one heavy.sh session | cuda12, `build.sh` (out12) | pass (13a) |
| 14 | The image on the GPU server: serve praxis and one curl on the GPU | GPU server, udocker 1.3.17 (P1, `setup --nvidia`) | the image from row 13, flattened | pass with workarounds (14a) |

### 1a. ubuntu:22.04, wget only (udocker on the GPU server)

```
== os: Ubuntu 22.04.5 LTS
== tools: /usr/bin/wget          libcurl.so.4 in the loader cache: 0
== install pinned 0.1.0
lernex: Linux x86_64: cpu build, the NVIDIA driver 595.91.07 is here, but nvidia-smi is not on PATH to check the GPU
lernex: downloading lernex-linux-x86_64-cpu.tar.gz
lernex: installed lernex 0.1.0 in ~/.lernex/versions/0.1.0
lernex: linked ~/.local/bin/lernex
lernex: libcurl is missing, so lernex cannot download models yet. Install it: sudo apt-get install -y curl ...
lernex: done. Try: lernex doctor
== version: lernex 0.1.0 (linux-x86_64-cpu, wip)
== doctor: CPU Intel Core i5-14500, 14 threads, AVX2 and AVX-VNNI; backend CPU, liblernex_cpu.so
== pull praxis (no libcurl)
lernex: libcurl is not installed, so models cannot be downloaded
== run praxis --think off, "What is 17 times 3?"
51    1 tokens, 30 tok/s
== upgrade (latest)
lernex: installed lernex 0.1.1 in ~/.lernex/versions/0.1.1 (was 0.1.0)
== version: lernex 0.1.1 (linux-x86_64-cpu, wip)   versions: 0.1.0 0.1.1   current -> versions/0.1.1
```

### 2a. ubuntu:24.04, curl installed (udocker on the GPU server)

The same sequence, with libcurl present (no warning). `lernex run praxis --think off` answered `51` at 32 tok/s, and
the re-run upgraded 0.1.0 to 0.1.1 (`versions: 0.1.0 0.1.1`, `current` at 0.1.1).

Notes for 1a and 2a:
- udocker's proot and fakechroot modes resolve `/proc/self/exe` to the dynamic loader, so the engine cannot find its
  backend next to the executable (`dispatch.cpp` derives the directory from that link). The phases set `LX_KERNELS`
  to the installed `liblernex_cpu.so`. That bypasses the probe, not the install: the tree, `bin/lernex --version`,
  the link and the CPU backend's own exports were used as installed.
- `doctor` lists the 15 `lx_program_*` symbols as missing on the CPU backend (see Findings).

### 10a. GPU server, native, install.sh from a local release host (05:00 UTC, 2026-10-28)

Run under `scripts/remote.sh installer x`, with `unset LX_KERNELS LX_BACKEND` in the command, so the installed tree
resolved its own backend. GPU: NVIDIA GeForce RTX 5070 Ti (sm_120), driver 595.91.07.

```
lernex: Linux x86_64: cuda12 build, NVIDIA GeForce RTX 5070 Ti, driver 595.91.07
lernex: downloading lernex-linux-x86_64-cuda12.tar.gz
lernex: installed lernex 0.1.0 in ~/.lernex/versions/0.1.0
lernex: linked ~/.local/bin/lernex
lernex: done. Try: lernex doctor
== version: lernex 0.1.0 (linux-x86_64-cuda12, 0609dd3)
== tree: ~/.lernex/versions/0.1.0, channel tarball
== doctor praxis: rc 1, "error: cannot open praxis/config.json" (the alias is not in this build, see Findings)
   the rest of doctor: GPU RTX 5070 Ti (device 0), sm_120, 16.61 GB; CUDA driver 13.2, runtime 12.8
   backend CUDA, liblernex_kernels.so in the installed tree; tuning from the installed tree
== run praxis --think off, "What is 17 times 3?"  rc 0:  51   1 tokens, 12 tok/s
== serve praxis --host 127.0.0.1 --port 18093: up in about 2 s
   GPU while serving: the engine's pid, 14,510 MiB (planned 14.77 GB of 14.78 GB)
== POST /v1/chat/completions (enable_thinking false): "51", finish_reason stop
== serve stopped
```

Logs: `~/installer-gpu/logs/row10/` on the GPU server (`install.txt`, `doctor.txt`, `run.txt`, `serve.log`,
`chat.json`); the row script is `~/installer-gpu/logs/row10.sh`.

### 13a. Image on this Mac (one heavy.sh session)

```
docker build --platform linux/amd64 -t lernex/engine:0.1.0-test \
  --build-arg LERNEX_VERSION=0.1.0 \
  --build-arg LERNEX_RELEASES=http://host.docker.internal:18243 \
  -f docker/Dockerfile docker
```

- Docker Desktop 29.4.0 started for the session (it was not running) with 8.3 GB and 10 CPUs. The base images were
  already in its cache, so the build did not pull them. The fetch stage downloaded the cuda12 tarball and its
  `SHA256SUMS` from the local release host and checked the tarball (`sha256sum -c`), then unpacked it to `/opt/lernex`.
- Build rc 0 in about 1 minute of build time (the x86_64 steps run under emulation).
- Image size (`docker image inspect`): 822,010,399 bytes.
- `docker run --rm --platform linux/amd64 --entrypoint lernex lernex/engine:0.1.0-test --version`:
  `lernex 0.1.0 (linux-x86_64-cuda12, 0609dd3)`.
- `docker save | gzip -3` to the GPU server: `~/installer-gpu/image/lernex-engine-0.1.0-amd64.tar.gz`, 818,419,493
  bytes, `gzip -t` ok.
- Docker Desktop was quit as soon as the save was verified. Quitting the GUI did not stop `com.docker.backend` within
  90 seconds, so the session stopped the backend at 05:04:01 and Docker was down at 05:04:06 (about 97 seconds after the
  quit started). The session log is `~/installer-gpu/logs/session.log`.

What this row does not show: `docker run` of the container with the GPU and its own `ENTRYPOINT` (serve on port 8080).
There is no Docker on the GPU server, so the same image was run under udocker in row 14 instead.

### 14a. The image on the GPU server (udocker 1.3.17)

The saved image was not loadable as it is, so the run used these steps:
1. `udocker load` fails on Docker 29's save format: with `index.json` present it reads the OCI index and raises
   `KeyError: 'index'`; with `index.json` removed it raises `KeyError: 'json'` on the layer metadata and `create` says
   "imagerepo is invalid".
2. The layers were flattened into one rootfs in the order of `manifest.json`, honouring whiteouts (8 layers, 1.5 GB,
   `flatten.py` in `~/installer-gpu/image/`), and imported with `udocker import`. The image's environment, entrypoint
   and command were not carried over; they were passed on the command line.
3. `udocker setup --execmode=P1` and `udocker setup --nvidia` added the host's NVIDIA libraries (libcuda 595.91.07,
   libnvidia-ml, and the rest) to the container.

Run with `udocker run` (P1), `-e LX_KERNELS=/opt/lernex/lib/liblernex_kernels.so`, and the models folder mounted at
`/root/.cache/lernex/models`:

```
doctor (model folder): rc 0; GPU RTX 5070 Ti; CUDA driver 13.2, runtime 12.8; backend CUDA, liblernex_kernels.so (/opt/lernex/lib)
run praxis --think off: rc 0, 51 at 12 tok/s
serve praxis --host 127.0.0.1 --port 8094: up; GPU compute app: /usr/local/bin/lernex, 14,510 MiB
POST /v1/chat/completions: "51"
serve stopped; the GPU is free afterwards (0%, 10 MiB)
```

Why R1 (the rootless runc mode) was not used: `udocker run` in R1 fails with
`nsexec: failed to unshare remaining namespaces (except cgroupns): Operation not permitted`. The server runs Ubuntu 26.04
with `kernel.apparmor_restrict_unprivileged_userns = 1`, which blocks the namespace setup for an unprivileged user. That
is a host setting for the owner to decide; it was not changed here.

Notes for 14a:
- `LX_KERNELS` is set because udocker's proot mode resolves `/proc/self/exe` to the loader, so the backend probe cannot
  find `lib/` next to the executable. The backend that ran is the image's own `liblernex_kernels.so` (the CUDA build).
- Logs: `~/installer-gpu/udocker-run/` (`doctor.txt`, `run.txt`, `serve.log`, `chat.json`, `log.txt`). The script is
  `~/installer-gpu/udocker-serve.sh`, run under `remote.sh x`.

### 6a. Driver and GPU selection (GPU server, shimmed `nvidia-smi` and `uname`)

| Case | Machine | Build chosen | Line printed |
|------|---------|--------------|--------------|
| A | no `nvidia-smi` on PATH, `/proc/driver/nvidia/version` says 595.91.07 | cpu | `the NVIDIA driver 595.91.07 is here, but nvidia-smi is not on PATH to check the GPU` |
| B | driver 470.182.03, sm_86 | cpu | `driver 470.182.03 is older than 525, the oldest CUDA build needs` |
| C | driver 535.183.01, sm_86 | cuda12 | `cuda12 build, NVIDIA GeForce RTX 3090, driver 535.183.01` |
| D | driver 595.91.07, sm_75 | cpu | `NVIDIA GeForce RTX 2080 is sm_75, and the CUDA builds need sm_80 or newer` |
| E | driver 595.91.07, sm_120 | cuda13 | `cuda13 build, NVIDIA GeForce RTX 5070 Ti, driver 595.91.07` |
| F | as E, release lists only cuda12 | cuda12 | `cuda12 build, NVIDIA GeForce RTX 5070 Ti, driver 595.91.07` |
| G | driver 595.91.07, compute capability query fails | cpu | `nvidia-smi cannot read the GPU's compute capability` |
| H | `LERNEX_FLAVOR=cuda13`, driver 535 | cuda13 | warning: `cuda13 needs an NVIDIA driver of 580 or newer ... the engine may not start` |
| I | `LERNEX_FLAVOR=cuda12`, release lists no cuda12 | refused | `this release has no lernex-linux-x86_64-cuda12.tar.gz, so there is no cuda12 build for this machine` |
| J | aarch64 (shimmed `uname -m`) | cpu | `Linux aarch64: cpu build` |
| K | `LERNEX_FLAVOR=cuda13` on aarch64 | refused | `LERNEX_FLAVOR does not apply on aarch64 Linux: the only build is cpu` |
| L | `LERNEX_FLAVOR=cuda99` | refused | `LERNEX_FLAVOR must be cpu, cuda12 or cuda13, not cuda99` |

Case A reads the real `/proc/driver/nvidia/version` on the GPU server, because no `nvidia-smi` was on that PATH. That
`/proc` parse was broken until this run caught it: the first number after "Kernel Module" is the `86` of `x86_64`.
The releases in these cases are fake; they exercise the choice, not the engine. The cuda13 branches stay in `install.sh`
(a release that ships cuda13 would use them); the release ships cuda12 only, so cases E and H need a release that has
cuda13 to mean anything.

### 7a. Install-link behaviour (this Mac, fake darwin tarballs 0.1.0, 0.1.1, 0.1.2, temp HOME)

- Fresh install 0.1.0: `current` is absolute and points at `versions/0.1.0`; no `previous`.
- Upgrade to 0.1.1: `previous` is absolute and points at `versions/0.1.0`; `versions` holds 0.1.0 and 0.1.1.
- Upgrade to 0.1.2: `versions` holds 0.1.1 and 0.1.2 (0.1.0 pruned); `previous` points at 0.1.1.
- Re-run of 0.1.2 with the `.install` marker: "already installed and current", nothing changes.
- Re-run of 0.1.2 with the marker removed (the state after `lernex update`): downloads, re-installs, writes the marker,
  and keeps `previous` at 0.1.1 instead of dropping the rollback target.
- Checksum mismatch: "does not match SHA256SUMS ... Nothing was installed"; no `versions` entry, no link moved.
- No `LERNEX_VERSION`: installs the latest, 0.1.2.

Before this change install.sh pointed `current` at a relative path and never wrote `previous`, so
`lernex update --rollback` (which reads `~/.lernex/previous` and checks the target as a directory from wherever it
runs) reported no previous version after an install.sh upgrade.

## Findings

- The CPU and Metal backends do not export the 15 `lx_program_*` symbols that `kernels.h` declares; `doctor` lists
  them missing on both. Pre-existing, not an installer problem.
- The cuda12 tarball (`0609dd3`) predates the `doctor` alias commit on main (`2ac61ea`): `doctor praxis` fails with
  "cannot open praxis/config.json" in row 10. A release rebuilt from main would fix it; the same tarball does accept a
  model folder path.
- `lernex update` does not write install.sh's `.install` marker. A re-run of install.sh after a `lernex update` to the
  same version downloads the release again and reinstalls it (harmless, slower). The release side could write the
  marker in `install_release`, or install.sh could compare `--version` output.
- Local `build.sh` runs without `LX_GIT_SHA` print `wip` in `--version`.
- udocker 1.3.17 cannot load a `docker save` archive from Docker 29 (see 14a). Flattening the layers works.
- On the GPU server, udocker's rootless runc mode is blocked by `kernel.apparmor_restrict_unprivileged_userns = 1`.
- In udocker's proot mode the backend probe needs `LX_KERNELS` (see 1a, 14a).
- Quitting Docker Desktop from the command line (`osascript quit`) left `com.docker.backend` running for 90 seconds; the
  session had to stop it. Docker should be quit from the GUI, or the session should wait longer before stopping it.
- A glibc below 2.34 needs `-pthread` on the link lines for the engine (CMake did not add it); `build.sh` runs in
  manylinux_2_28, which has the newer glibc, so the release is unaffected.

## Docker

- `docker/Dockerfile` (on main): multi-stage. A plain Ubuntu 24.04 stage fetches the flavor's tarball, checks it against
  `SHA256SUMS` and unpacks it to `/opt/lernex`; the final stage is NVIDIA's CUDA base. The default is cuda12 on
  `nvidia/cuda:12.8.1-base-ubuntu24.04`, the CUDA 12.8 the release ships; the base tag exists on Docker Hub. The tarball
  bundles the CUDA runtime, so the base needs no toolkit.
- Entrypoint: `lernex serve --host 0.0.0.0 --port 8080`, default command `praxis`. A deviation from the brief's
  `ENTRYPOINT ["lernex","serve"]` plus `CMD` flags: `docker run image praxis` serves on port 8080 without repeating the
  flags, and `--entrypoint lernex` still reaches the other commands.
- `.github/workflows/docker.yml`: runs when a GitHub release is published (the owner publishes the draft from
  `release.yml`), not on the tag push. A tag push starts before the draft is published, so the image would wait on
  assets that do not exist yet. A manual run (`workflow_dispatch` with a version) builds the image and pushes nothing.
  `packages: write` is set on the job for the GHCR push. `:latest` is not set for a prerelease. The flavor is cuda12.
- Not verified: `docker run` of the image itself (with the GPU, or its own entrypoint). Row 13 covers the build and the
  `--version` check; row 14 covers the same tree under udocker.

## Open items

- Rows 4, 11 and 12 are not run in this round. Row 4 (darwin native install) is a light test on this Mac once a darwin
  build is ready. Rows 11 and 12 need Docker Desktop on the Mac for a second session; they were left out so the image
  session stayed alone on the machine.
- Row 14 with Docker's own GPU path needs Docker with the NVIDIA Container Toolkit on a Linux GPU host (owner). udocker
  R1 needs `kernel.apparmor_restrict_unprivileged_userns` relaxed on the server (owner).
- The release tarball behind rows 10 and 13 needs a rebuild from main for the `doctor` alias.
