# Install tests

What `scripts/install.sh`, the Docker image, `lernex update` and the install paths were run on, with the result of
each run. Every row names the machine, the release build it installed and what happened. There have been two rounds:
the release candidate v0.1.0-rc1 (round 2, first below) and the 0.1.0 test builds before it (round 1). A row marked
"not run" was not run in its round; the Open items say what is left.

## Round 2: v0.1.0-rc1 (2026-10-28)

The tag `v0.1.0-rc1` is on `7608083`. Every row in this round installed the release candidate's own tarballs, built
from the tag, through the tag's own `install.sh` and `docker/Dockerfile`.

### How the release candidate was built

The tag's release workflow (run 40783732071) did not start: GitHub refuses Actions jobs on the private repository
until the account's included minutes reset on November 1 (they ran out on October 19, and the spending limit is $0).
Public repositories run free, so the v0.1.0 tag, pushed once the repository is public, runs `release.yml` as written.

So the four tarballs were built the way `release.yml` builds them, from `git archive v0.1.0-rc1`, with the workflow's
own scripts and base images, on our machines:

| Tarball | Where | How | Bytes | sha256 |
|---------|-------|-----|------:|--------|
| `lernex-linux-x86_64-cpu.tar.gz` | GPU server, Docker | `quay.io/pypa/manylinux_2_28_x86_64`, `scripts/release/build.sh`, 2 min at `LX_JOBS=4` | 3,583,197 | `3b5885e4…` |
| `lernex-linux-x86_64-cuda12.tar.gz` | GPU server, Docker | the same base, `scripts/release/cuda-toolkit.sh 12.8`, then `build.sh`, 17 min at `LX_JOBS=4` | 563,814,407 | `36bdfee3…` |
| `lernex-linux-aarch64-cpu.tar.gz` | this Mac, Docker Desktop, linux/arm64 | `quay.io/pypa/manylinux_2_28_aarch64`, `build.sh` (GCC 14.2.1) | 3,007,222 | `852ed9f5…` |
| `lernex-darwin-arm64.tar.gz` | this Mac, native, under `heavy.sh` | `build.sh` (AppleClang 21) | 1,248,377 | `6db3f583…` |

- Every build had `LX_VERSION=0.1.0-rc1` and `LX_GIT_SHA=7608083`, as the workflow sets them from the tag, so
  `--version` prints `lernex 0.1.0-rc1 (FLAVOR, 7608083)` in every row.
- `SHA256SUMS` was written by `scripts/release/sums.sh`. `install.sh` is the tag's, sha256 `e0fd2436…`, the same file
  round 1 tested.
- The CUDA build leaves the decode program out (`LX_DECODE_PROGRAM` is off in the release flavors, `80658f7`). The
  tarball is 564 MB against round 1's 713 MB. Unpacked it is 886 MB, of which `libcublasLt.so.12` is 752 MB and
  `liblernex_kernels.so` 129 MB.
- The release host is round 1's: a folder shaped like GitHub's releases, served on 127.0.0.1 (port 18240 on the GPU
  server, 18250 on this Mac, reached from Docker Desktop containers as `host.docker.internal`). Row 15 adds GitHub's
  `/latest` redirect (see 15a).
- The scripts are `~/installer-rc/` on the GPU server (`build-rc.sh`, `rc-tests.sh`, `update-test.sh`, `relhost.py`,
  logs in `out-0.1.0-rc1/` and `out-update/`). The Mac rows ran from a session scratch folder with the same layout.

### Results

| # | What | Where | Result |
|---|------|-------|--------|
| 4 | `install.sh` (latest), native: `--version`, `doctor`, signature, quarantine, re-run | this Mac, macOS 27.0, Apple M5, temp HOME | pass (4b) |
| 10 | `install.sh` (latest), native, cuda12 chosen by itself: `--version`, `doctor praxis`, one prompt, `serve` and one curl | GPU server, RTX 5070 Ti, driver 595.91.07, temp HOME | pass (10b) |
| 11 | arm64 Ubuntu 22.04 and 24.04: install, `--version`, `doctor`, re-run | this Mac, Docker Desktop 29.4.0, linux/arm64 | pass (11b) |
| 12 | x86_64 Ubuntu 22.04 (wget only) and 24.04 (curl): install, `--version`, `doctor`, one prompt on the CPU, re-run | GPU server, Docker 29.1.3, native x86_64 (round 1 planned emulation on the Mac) | pass (12b) |
| 13 | Image build from the tag's `docker/Dockerfile` against the release candidate's host | GPU server, Docker 29.1.3 | pass (13b) |
| 14 | The image through Docker's GPU path: `docker run --gpus all`, its own ENTRYPOINT and CMD, one curl, `doctor` | GPU server, NVIDIA Container Toolkit | pass (14c) |
| 15 | `lernex update` end to end: 0.1.0-rc1 installed pinned, `update --check`, `update` to 0.1.0, check again, `--rollback`; the daily notice in `lernex run` | GPU server, ubuntu:24.04 container, a test host with GitHub's `/latest` redirect | pass (15a) |

Rows 5 to 9 test `install.sh`'s choices and refusals with shims and fake tarballs. `install.sh` is byte for byte the
file they passed on, so they were not rerun.

### 4b. This Mac, native (macOS 27.0, Apple M5, temp HOME)

```
lernex: done. Try: lernex doctor                        install rc 0
version    lernex 0.1.0-rc1 (darwin-arm64, 7608083)
doctor     rc 0; Metal driver 27.0, runtime 4.0; backend Metal, liblernex_metal.dylib in the installed tree; missing none
signature  Signature=adhoc, TeamIdentifier not set
quarantine no com.apple.quarantine on ~/.lernex/current/bin/lernex
re-run     "lernex 0.1.0-rc1 is already installed and current", rc 0
```

`doctor` on this build still printed CUDA's fields for the Apple GPU (`arch sm_1010`, `L2 0 MB`, `clusters yes`). Main
fixed that after the tag (see Findings).

### 10b. GPU server, native (RTX 5070 Ti, driver 595.91.07, temp HOME)

```
lernex: Linux x86_64: cuda12 build, NVIDIA GeForce RTX 5070 Ti, driver 595.91.07
lernex: installed lernex 0.1.0-rc1 in ~/.lernex/versions/0.1.0-rc1
version       lernex 0.1.0-rc1 (linux-x86_64-cuda12, 7608083)
doctor praxis rc 0: sm_120; CUDA driver 13.2, runtime 12.8; kernels for sm_80 sm_86 sm_89 sm_90 sm_100 sm_120;
              backend CUDA, liblernex_kernels.so in the installed tree; missing none; tuning from the installed tree;
              the plan for praxis (BF16 stored and chosen, BF16X on, speculation on for one stream)
run praxis --think off, "What is 17 times 3?": 51
serve praxis on 127.0.0.1:18241: up to 30 sequences of 8192 tokens, 14.68 GB planned of 14.78 GB
POST /v1/chat/completions: "51", finish_reason stop
```

`doctor praxis` failing on the model alias was round 1's finding; it passes here. The run and serve held the GPU lock
and waited about five minutes for the engine work ahead of them.

### 11b. arm64 Ubuntu (Docker Desktop on this Mac)

Both containers: install rc 0, `lernex 0.1.0-rc1 (linux-aarch64-cpu, 7608083)`, `doctor` rc 0 (Advanced SIMD, dot
product, I8MM, BF16; 116 GB/s measured on 10 threads; backend `liblernex_cpu.so`; missing none), and a re-run that
printed "already installed and current". Docker Desktop was started for the row and quit after it.

### 12b. x86_64 Ubuntu (Docker on the GPU server)

```
22.04, wget only: install rc 0, with the libcurl warning; lernex 0.1.0-rc1 (linux-x86_64-cpu, 7608083)
                  doctor: i5-14500, AVX2, FMA, F16C, AVX-VNNI; 74 GB/s measured on 14 threads; missing none
                  run praxis --think off on the CPU: 51, 19 tok/s
24.04, curl:      install rc 0; lernex 0.1.0-rc1 (linux-x86_64-cpu, 7608083); doctor as 22.04
                  run praxis --think off on the CPU: 51, 26 tok/s
                  re-run, both: "lernex 0.1.0-rc1 is already installed and current"
```

The rate is one generated token over its own time, so it says little, and this run shared the CPU with row 15's
notice container.

### 13b and 14c. The image, on the GPU server

Built from the tag's `docker/Dockerfile` with Docker 29.1.3, on the host network so that the fetch stage reached the
release candidate's host:

    docker build --network host -t lernex/engine:0.1.0-rc1 \
      --build-arg LERNEX_VERSION=0.1.0-rc1 --build-arg LERNEX_RELEASES=http://127.0.0.1:18240 .

```
fetch stage   lernex-linux-x86_64-cuda12.tar.gz: OK (sha256sum -c against the release's SHA256SUMS)
build         rc 0 in 3.6 minutes, nvidia/cuda:12.8.1-base-ubuntu24.04 pulled fresh
image         670,718,068 bytes (round 1's 0609dd3 image: 822,010,399)
config        ENTRYPOINT lernex serve --host 0.0.0.0 --port 8080, CMD praxis
--version     lernex 0.1.0-rc1 (linux-x86_64-cuda12, 7608083)
```

Then the image's own ENTRYPOINT and CMD, the GPU through `--gpus all`, the models folder mounted, under the GPU lock:

    docker run -d --rm --gpus all -p 127.0.0.1:18242:8080 -v ~/.cache/lernex/models:/root/.cache/lernex/models lernex/engine:0.1.0-rc1

```
serve         Serving praxis on http://0.0.0.0:8080/v1: up to 30 sequences of 8192 tokens, 14.68 GB planned of 14.78 GB
POST /v1/chat/completions ("What is 17 times 3?", thinking off): "51", finish_reason stop
doctor        (docker exec, while serving) sm_120; CUDA driver 13.2, runtime 12.8; kernels for sm_80 sm_86 sm_89 sm_90 sm_100 sm_120;
              backend CUDA, /opt/lernex/lib/liblernex_kernels.so; missing none; tuning from the image
stopped       docker stop; no compute apps on the GPU afterwards
```

The image was built on the GPU server, not saved and copied from this Mac as in round 1, now that the server has
Docker. It stays there as `lernex/engine:0.1.0-rc1` and is not pushed anywhere.

### 15a. `lernex update` end to end

`lernex update` asks the release host for `/latest` and reads the version from where it redirects
(`/releases/tag/vX.Y.Z` on GitHub). A plain file server has no such redirect, and the release candidate is a
prerelease, which `update` never offers (GitHub's `/releases/latest` never names one). So this row adds what the real
host has:

- `relhost.py` serves the folder and answers `/latest` with a 302 to `/tag/v0.1.0`, as GitHub does.
- A test-only 0.1.0 x86_64 CPU tarball, built from the same `v0.1.0-rc1` source with `LX_VERSION=0.1.0` in the same
  manylinux base, plays the newer release. It was never published and exists only on the GPU server.

In a fresh ubuntu:24.04 container on the GPU server, with `LERNEX_RELEASES` pointing at that host:

```
install 0.1.0-rc1 pinned        lernex: done. Try: lernex doctor
version                         lernex 0.1.0-rc1 (linux-x86_64-cpu, 7608083)
update --check                  lernex 0.1.0 is available (you have 0.1.0-rc1). Run `lernex update` to install it.
run praxis --think off          51
update                          downloading lernex 0.1.0 (linux-x86_64-cpu)
                                lernex 0.1.0 is installed and current. lernex 0.1.0-rc1 is kept: `lernex update --rollback` returns to it.
version after update            lernex 0.1.0 (linux-x86_64-cpu, 7608083)
~/.lernex/versions              0.1.0 0.1.0-rc1, current -> versions/0.1.0
update --check again            lernex 0.1.0 is the latest release.
update --rollback               lernex switched back to 0.1.0-rc1. The next run uses it.
version after rollback          lernex 0.1.0-rc1 (linux-x86_64-cpu, 7608083)
```

The daily notice, in a second fresh container against the same host. `lernex run` prints it from
`~/.cache/lernex/update-check`, which the previous run's background check writes, so it takes two runs to appear:

```
run 1 (no cache yet)     51, no notice; afterwards the cache reads latest=0.1.0
run 2                    51, and on stderr: lernex 0.1.0 is out (you have 0.1.0-rc1). Run `lernex update` to install it.
update                   to 0.1.0
run 3                    51, no notice
```

Against the plain hosts of rows 4 and 10 to 12, which have no `/latest` redirect, `update --check` printed "the
release host names no latest release", as it should. On 22.04 without libcurl it printed "libcurl is not available
on this system", matching the installer's warning.

## Round 1 method: the 0.1.0 test builds

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

## Round 1 results

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
| 14b | The image on the GPU server through Docker's own GPU path: `docker run --gpus all`, the image's ENTRYPOINT and CMD, one curl | GPU server, Docker 29.1.3 with the NVIDIA Container Toolkit | the image from row 13, `docker load` as saved | pass (14b) |

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
There was no Docker on the GPU server at the time, so the same image was run under udocker in row 14; 14b later ran it
through Docker itself.

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

### 14b. The image on the GPU server through Docker (2026-10-28)

The owner installed docker.io 29.1.3 and the NVIDIA Container Toolkit on the GPU server (runtimes runc and nvidia).
This is the path people will use: the saved image from row 13 loads as it is (`docker load`, no flattening), and runs
with its own ENTRYPOINT (`lernex serve --host 0.0.0.0 --port 8080`) and CMD (`praxis`), the GPU through `--gpus all`
and the models folder mounted, with no `LX_KERNELS` or other workaround:

    docker run -d --gpus all -p 127.0.0.1:8094:8080 -v ~/.cache/lernex/models:/root/.cache/lernex/models lernex/engine:0.1.0-test

```
--version: lernex 0.1.0 (linux-x86_64-cuda12, 0609dd3)
/health: up after 10 polls at 2 s
GPU compute app: lernex, 14,510 MiB
POST /v1/chat/completions ("What is 17 times 3?", thinking off): rc 0, "51"
doctor (docker exec): RTX 5070 Ti sm_120; CUDA driver 13.2, runtime 12.8; backend CUDA, liblernex_kernels.so (/opt/lernex/lib); missing none
serve: up to 31 sequences of 8192 tokens; about 290 tok/s at one stream planned
stopped (docker stop); the GPU is free afterwards (0%, 10 MiB)
```

The run held the GPU lock. Script and logs: `~/installer-gpu/docker-run/` (`serve.sh`, `version.txt`, `log.txt`,
`chat.json`, `doctor.txt`, `serve.log`). The image is the row 13 build at 0609dd3, so the release candidate's image
gets the same run once it is built from the RC tarball.

## Findings

- Resolved in rc1: `doctor` listed the 15 `lx_program_*` symbols as missing on the CPU and Metal backends. The decode
  program is CUDA's alone and opt-in, so `doctor` leaves those names out, and rc1 prints `missing none` on all three
  backends (rows 4, 10 to 12).
- Resolved in rc1: round 1's cuda12 tarball (`0609dd3`) predated the `doctor` alias, so `doctor praxis` failed with
  "cannot open praxis/config.json". rc1's `doctor praxis` passes (10b).
- Fixed on main after rc1 (`80bfefd`): on a Mac, `doctor` printed CUDA's fields for the Apple GPU (`arch sm_1010`,
  `SMs 10, L2 0 MB`, `tensor cores`, `clusters yes (sm_90 and up)`). It is the first command `install.sh` suggests.
  It now prints the Apple family, the GPU cores, the unified memory the GPU can use, whether the GEMMs and attention run
  on the neural accelerators (Apple family 10 and later, `LX_METAL_NAX=0` turns them off), and the macOS and Metal
  versions. Checked on this Mac with a darwin build of main, with and without `LX_METAL_NAX=0` and
  `LX_ARCH_OVERRIDE=1009`. v0.1.0 carries it; rc1's darwin tarball does not.
- The darwin binary is ad-hoc signed (no Developer ID, not notarized). `install.sh` downloads with curl, which sets no
  quarantine, so the install runs (4b); Homebrew's formula downloads with curl too. A tarball downloaded in a browser
  is quarantined, and Gatekeeper refuses an ad-hoc binary from it. Signing and notarizing in `release.yml` needs a
  Developer ID Application certificate and a notary credential as repository secrets.
- `lernex update` does not write install.sh's `.install` marker. A re-run of install.sh after a `lernex update` to the
  same version downloads the release again and reinstalls it (harmless, slower). The release side could write the
  marker in `install_release`, or install.sh could compare `--version` output.
- Local `build.sh` runs without `LX_GIT_SHA` print `wip` in `--version`. The release workflow sets it, and the rc1
  builds set it the same way.
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
- Verified: `docker run --gpus all` of the image with its own entrypoint and default command, serve and one chat
  request on the GPU (round 1's 14b, and round 2's 14c on the release candidate's image). Rows 13a and 13b cover the
  build and the `--version` check.

## Open items

- `release.yml`, `pypi.yml` and `docker.yml` have not run on GitHub yet: Actions is held on the private repository by
  billing until November 1 (round 2, How the release candidate was built). Their first run is the v0.1.0 tag once the
  repository is public. Round 2 followed `release.yml`'s steps by hand on the same base images and scripts, and
  `actionlint` 1.7.7 passes on all three workflows apart from one intentional SC2086 in `pypi.yml` (`$CUDA_FLAG` is
  left unquoted so that an empty flag adds no argument).
- Signing and notarizing the darwin tarball needs a Developer ID Application certificate and a notary credential as
  repository secrets (Findings). Until then `install.sh` works on a Mac and a browser-downloaded tarball does not.
- Rows 5 to 9 were not rerun in round 2, because `install.sh` has not changed since they passed. Rerun them whenever it
  does.
- udocker's R1 mode is not pursued: the GPU server keeps `kernel.apparmor_restrict_unprivileged_userns = 1` on
  purpose, and Docker with the NVIDIA runtime replaces udocker there (14b, and round 2's row 14).
