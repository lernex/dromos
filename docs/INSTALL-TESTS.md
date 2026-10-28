# Install tests

What `scripts/install.sh`, the Docker image and the install paths were run on, with the result of each run. Every row
names the machine, the release build it installed and what happened. A row marked pending has not run yet; the
Open items say what it waits for.

## Method

- Release host: a directory laid out like GitHub's (`latest/download/` and `download/vX.Y.Z/`, each holding the
  tarballs, `SHA256SUMS` and `install.sh`), reached through `LERNEX_RELEASES`. `SHA256SUMS` is written by
  `scripts/release/sums.sh`, as the release workflow does.
- The tarballs are the release agent's `scripts/release/build.sh` outputs, not hand-assembled trees, wherever a row
  says `build.sh`. Their `--version` reads `wip` because those local builds had no `LX_GIT_SHA`; the release
  workflow sets it from the commit.
- Each run starts from a fresh `HOME` or a fresh container, so the install is the first install on the machine.
- The Praxis weights come from the GPU server's cache. Containers mount a hard-linked copy of that folder, so no
  6 GB download happens in a test.
- Flags that were needed to get a run through, and why, are in the Notes.

## Results

| # | What | Where | Build | Result |
|---|------|-------|-------|--------|
| 1 | Fresh install 0.1.0, `--version`, `doctor`, `pull` without libcurl, one prompt, upgrade to 0.1.1 | ubuntu:22.04 x86_64, udocker on the GPU server, wget only | cpu, `build.sh` (out/v010, out/v011) | pass (1a) |
| 2 | Same as 1 | ubuntu:24.04 x86_64, udocker on the GPU server, curl | cpu, `build.sh` | pass (2a) |
| 3 | Generation, one prompt, thinking off | rows 1 and 2 | cpu | pass: "17 times 3" answered `51`, 30 and 32 tok/s |
| 4 | Darwin install, `--version`, `doctor`, Metal backend loads, temp HOME | this Mac, arm64 | metal, `build.sh` at HEAD | pending (Mac session, 4b) |
| 5 | Install refusals: Intel Mac, Rosetta shell, macOS 13 | this Mac, shimmed `uname`, `sysctl`, `sw_vers` | n/a | pass, each prints its reason and installs nothing (run before the link change; the refusals come before it) |
| 6 | Driver and GPU selection, cases A to L | GPU server, shimmed `nvidia-smi` and `uname` | fake tarballs | pass, rerun on the current `install.sh`, same outcomes as 6a |
| 7 | Upgrade, re-install with and without `.install`, previous link, piped install | this Mac, temp HOME | fake darwin tarballs | pass (7a) |
| 8 | Piped install, tty progress bar, idempotent PATH line, fish config | this Mac | darwin | pass (earlier run, before the link change) |
| 9 | Checksum mismatch, bad version, bad `LERNEX_FLAVOR`, a non-Lernex file at `~/.local/bin/lernex` | this Mac and GPU server | fake | pass: nothing installed in each, rerun on the current `install.sh` |
| 10 | GPU install (cuda13 chosen by itself), `doctor`, one prompt with `lernex run`, `serve` and `/v1/chat/completions` | GPU server, native, temp HOME | cuda13 | queued, see Open items |
| 11 | arm64 Ubuntu 22.04 and 24.04: install, `doctor`, upgrade | Docker Desktop, linux/arm64 | cpu, `build.sh` (aarch64) | pending (Mac session, 4b) |
| 12 | amd64 Ubuntu 22.04 and 24.04 under emulation: install, `doctor`, upgrade | Docker Desktop, linux/amd64 | cpu, `build.sh` (x86_64) | pending (Mac session, 4b) |
| 13 | Image build `linux/amd64` from `docker/Dockerfile`, `--version` and `doctor` in the image | this Mac, buildx | cuda13, `build.sh` | pending (needs the cuda13 build and the Docker session) |
| 14 | Image on the GPU server | n/a | n/a | not possible here: the server has no Docker, see Open items |

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

Real-tarball inputs: `lernex-linux-x86_64-cpu.tar.gz` 0.1.0 (sha256 `beacf016bea59db1…`) and 0.1.1 (`b4601a5435e1…`),
from the release agent's `build.sh` output on the GPU server, served with `SHA256SUMS` from `sums.sh`.

Notes for 1a and 2a:
- udocker's proot and fakechroot modes resolve `/proc/self/exe` to the dynamic loader, so the engine cannot find its
  backend next to the executable (`dispatch.cpp` derives the directory from that link). The phases set `LX_KERNELS`
  to the installed `liblernex_cpu.so`. That bypasses the probe, not the install: the tree, `bin/lernex --version`,
  the link and the CPU backend's own exports were used as installed.
- `doctor` lists the 15 `lx_program_*` symbols as missing on the CPU backend (see Findings).

### 4b. Mac, containers and the native darwin install

Pending: the Mac session (the installer's scratchpad `docker2/session.sh`, run under `heavy.sh`) installs the
`build.sh` darwin tarball into a temp HOME, runs `--version` and `doctor`, and runs the arm64 and amd64 container
rows. This section is filled in from that run.

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
The releases in these cases are fake; they exercise the choice, not the engine.

## Findings for the release side

- The CPU and Metal backends do not export the 15 `lx_program_*` symbols that `kernels.h` declares; `doctor` lists
  them missing on both. Pre-existing, not an installer problem.
- Local `build.sh` runs without `LX_GIT_SHA` print `wip` in `--version`.
- `lernex update` does not write install.sh's `.install` marker. A re-run of install.sh after a `lernex update` to the
  same version downloads the release again and reinstalls it (harmless, slower). The release side could write the
  marker in `install_release`, or install.sh could compare `--version` output.
- A glibc below 2.34 needs `-pthread` on the link lines for the engine (CMake did not add it); `build.sh` runs in
  manylinux_2_28, which has the newer glibc, so the release is unaffected.

## Docker

- `docker/Dockerfile`: multi-stage. A plain Ubuntu 24.04 stage fetches the flavor's tarball, checks it against
  `SHA256SUMS` and unpacks it to `/opt/lernex`; the final stage is NVIDIA's CUDA base (`13.4.3-base-ubuntu24.04` by
  default, the 13.4 toolkit the release builds with; `12.8.1-base-ubuntu24.04` for cuda12, passed as `CUDA_BASE`).
  Both tags were checked against Docker Hub. The tarball bundles the CUDA runtime, so the base needs no toolkit.
- Entrypoint: `lernex serve --host 0.0.0.0 --port 8080`, default command `praxis`. A deviation from the brief's
  `ENTRYPOINT ["lernex","serve"]` plus `CMD` flags: `docker run image praxis` serves on port 8080 without repeating the
  flags, and `--entrypoint lernex` still reaches the other commands.
- `.github/workflows/docker.yml`: runs when a GitHub release is published (the owner publishes the draft from
  `release.yml`), not on the tag push. A tag push starts before the draft is published, so the image would wait on
  assets that do not exist yet. A manual run (`workflow_dispatch` with a version) builds the image and pushes nothing.
  `packages: write` is set on the job for the GHCR push. `:latest` is not set for a prerelease.

## Open items

- Row 10 (GPU install): the GPU is held by an external vLLM process (100%, 15.7 GB of 16.3 GB) that is not under the
  GPU lock, and five GPU jobs were queued ahead on `~/lernex-engine-dev/.gpu.lock`. The job runs
  `~/installer-test/gpu-test.sh` from `~/installer-gpu/rel`; that release host holds a stand-in cuda13 tarball until the
  release agent's real one is staged there.
- Rows 4, 11 to 13 (Mac): the Mac heavy queue was behind other agents' builds (an aarch64 release build, a darwin
  rebuild, and a `brew test` holding the lock for 40 minutes at the time of writing). The session script is ready.
- Row 14: the GPU server has no Docker and no root. Running the image with the GPU needs Docker with the NVIDIA
  Container Toolkit on a Linux GPU host (owner).
