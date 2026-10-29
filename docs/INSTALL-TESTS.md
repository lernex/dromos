# Install tests

What `scripts/install.sh`, the Docker image, `dromos update` and the install paths were run on, with the result of
each run. Every row names the machine, the release build it installed and what happened. There have been four
rounds, newest first: the Linux and macOS channels under the Dromos names (round 4, 0.1.0-rc2), Windows (round 3),
the release candidate v0.1.0-rc1 (round 2) and the 0.1.0 test builds before it (round 1). A row marked "not run" was
not run in its round; the Open items say what is left.

The engine was renamed Dromos on 2026-10-29, after rounds 1 and 2. They ran under the old name, so their commands and
output say `lernex`. In today's names: the command is `dromos`, `LERNEX_*` is `DROMOS_*`, `~/.lernex` is `~/.dromos`,
the assets are `dromos-*.tar.gz`, and the backends are `libdromos_*`.

## Round 4: 0.1.0-rc2, Linux and macOS under the Dromos names (2026-10-29)

Every Linux and macOS channel again, after the rename and before the freeze: `install.sh`, the image, `dromos
update`, and for the first time Homebrew and pip. `0.1.0-rc2` is the version the builds were given, not a tag: Actions
is still held on the private repository (round 2), so the builds followed `release.yml` by hand, as in round 2.

### How it was built

| Tarball | Where | How | Bytes | sha256 |
|---------|-------|-----|------:|--------|
| `dromos-linux-x86_64-cpu.tar.gz` | GPU server, Docker | `quay.io/pypa/manylinux_2_28_x86_64`, `scripts/release/build.sh`, 1 min at `LX_JOBS=4` | 3,588,035 | `932bdda3…` |
| `dromos-linux-x86_64-cuda12.tar.gz` | GPU server, Docker | the same base, the source's own `cuda-toolkit.sh 12.8`, then `build.sh`, 18 min at `LX_JOBS=4` | 565,240,808 | `dd7fc551…` |
| `dromos-linux-aarch64-cpu.tar.gz` | this Mac, Docker Desktop, linux/arm64 | `quay.io/pypa/manylinux_2_28_aarch64`, `build.sh` | 3,010,702 | `65030a86…` |
| `dromos-darwin-arm64.tar.gz` | this Mac, native, under `heavy.sh` | `build.sh` | 1,250,049 | `baa3aabd…` |

- The CPU and darwin tarballs are from `4a85ea3`, the rename with its release address fixed (Findings). The cuda12
  tarball is from `be70353`, which is `95e43a2` plus the toolkit change that landed as `6822467`, because the CUDA
  build of `4a85ea3` failed twice:
  - NVIDIA's RHEL 8 and 9 repository indexes named files that were not there from 03:13 UTC on October 29, which
    stopped the build at `dnf`. `cuda-toolkit.sh` now installs from NVIDIA's redistributable archives, pinned to
    `redistrib_12.8.2.json` and its sha256 with every archive checked against it: nvcc 12.8.93, cudart and CCCL
    12.8.90, cuBLAS 12.8.5.5, the versions rc1 got from the repository.
  - `4a85ea3`'s skinny GEMM did not compile for sm_80, 86 and 89 (its cluster calls). `95e43a2` fixed it, and every
    architecture has been compiled with each benchmark table since.
- `--version` prints `dromos 0.1.0-rc2 (FLAVOR, 4a85ea3)`, or `be70353` for cuda12. `install.sh` is `4a85ea3`'s,
  sha256 `2b1416aa…`.
- The cuda12 tree unpacks to 901 MB: `libcublasLt.so.12` 752 MB, `libdromos_kernels.so` 140 MB (rc1: 129 MB).
- The release hosts are plain folders shaped like GitHub's, on 127.0.0.1:18370 on the GPU server and 18250 on this
  Mac, plus `relhost.py` with GitHub's `/latest` redirect on 18380 for row 15. The scripts are `~/installer-rc3/` on
  the GPU server (`build-rc.sh`, `build-tk.sh`, `rc-tests.sh`, logs in `out-0.1.0-rc2/`); the Mac rows ran from a
  session scratch folder with the same layout.

### Results

| # | What | Where | Result |
|---|------|-------|--------|
| 4 | `install.sh` (latest), native: `--version`, `doctor`, signature, quarantine, re-run | this Mac, macOS 27.0, Apple M5, temp HOME | pass (4c) |
| 10 | `install.sh` (latest), native, cuda12 chosen by itself: `--version`, `doctor praxis`, one prompt, `serve` and one curl | GPU server, RTX 5070 Ti, driver 595.91.07, temp HOME | pass (10c) |
| 11 | arm64 Ubuntu 22.04 and 24.04: install, `--version`, `doctor`, re-run | this Mac, Docker Desktop 29.4.0, linux/arm64 | pass (11c) |
| 12 | x86_64 Ubuntu 22.04 (wget only) and 24.04 (curl): install, `--version`, `doctor`, one prompt on the CPU | GPU server, Docker 29.1.3 | pass (12c) |
| 13 | Image build from `4a85ea3`'s `docker/Dockerfile` against this round's host | GPU server, Docker 29.1.3 | pass (13c) |
| 14 | The image through Docker's GPU path: `docker run --gpus all`, its own ENTRYPOINT and CMD, one curl, `doctor`, `update --check` | GPU server, NVIDIA Container Toolkit | pass (13c) |
| 15 | `dromos update` end to end and the daily notice: 0.1.0-rc2 pinned, two runs, `update` to 0.1.0, a third run, `--rollback` | GPU server, ubuntu:24.04 container, `relhost.py` | pass (15b) |
| 25 | Homebrew: the formula `bump-formula.sh` writes for this release, in a local tap; install, `--version`, `doctor`, `update --check`, `brew test`, `brew audit --strict`, `brew style`, one prompt on Metal, uninstall and untap | this Mac, Homebrew 7.0.11 | pass (25a) |
| 26 | pip on macOS: `build_wheel.py` from the darwin tarball, `twine check`, install in a fresh venv, `--version`, `doctor`, `update --check`, one prompt on Metal | this Mac, Python 3.12.13 | pass (25a) |
| 27 | pip on aarch64 Linux: the `manylinux_2_28_aarch64` wheel, `twine check`, install, `--version`, `doctor`, `update --check` | this Mac, Docker Desktop, Ubuntu 24.04, Python 3.12.3 | pass (27a) |
| 28 | pip with CUDA: the cuda12 wheel as `pypi.yml` builds it, in a fresh venv on the GPU: `--version`, `doctor praxis`, one prompt, `update --check` | GPU server, Python 3.14, temp venv | pass (28a) |
| 29 | The same wheel in Ubuntu 24.04 with no GPU: install, `--version`, `doctor`, one prompt on the CPU | GPU server, Docker 29.1.3 | pass (28a) |

Rows 5 to 9 test `install.sh`'s choices and refusals with shims and fake tarballs, and were not rerun. Since rc1,
`install.sh` changed only by the rename: rc1's file with `LERNEX` made `DROMOS` and `lernex` made `dromos` is this
round's file line for line, apart from the release address (Findings). Rows 25 to 29 are new; 16 to 24 are round 3's.

### 4c, 11c and 12c. `install.sh` on this Mac, arm64 Ubuntu and x86_64 Ubuntu

```
4c   this Mac      install rc 0; dromos 0.1.0-rc2 (darwin-arm64, 4a85ea3)
                   doctor rc 0: Apple M5, Apple 10, 10 GPU cores, 12.71 GB unified memory the GPU can use,
                   neural accelerators on, macOS 27.0, Metal 4; backend libdromos_metal.dylib; missing none
                   Signature=adhoc, TeamIdentifier not set; no com.apple.quarantine
                   re-run: "dromos 0.1.0-rc2 is already installed and current"
11c  arm64 22.04   install rc 0; dromos 0.1.0-rc2 (linux-aarch64-cpu, 4a85ea3)
     arm64 24.04   the same; doctor: Advanced SIMD, dot product, I8MM, BF16, 101 GB/s measured on 10 threads;
                   missing none; re-run "already installed and current"
12c  x86_64 22.04  wget only: install rc 0; dromos 0.1.0-rc2 (linux-x86_64-cpu, 4a85ea3); run praxis on the CPU: 51
     x86_64 24.04  curl: the same; 51
```

`doctor` on the Mac prints the Apple GPU's own fields, the fix round 2 recorded after rc1.

### 10c. GPU server, native (RTX 5070 Ti, driver 595.91.07, temp HOME)

```
dromos: Linux x86_64: cuda12 build, NVIDIA GeForce RTX 5070 Ti, driver 595.91.07
version       dromos 0.1.0-rc2 (linux-x86_64-cuda12, be70353)
doctor praxis rc 0: sm_120; CUDA driver 13.2, runtime 12.8; kernels for sm_80 sm_86 sm_89 sm_90 sm_100 sm_120;
              backend CUDA, libdromos_kernels.so in the installed tree; missing none; tuning from the installed tree
run praxis --think off, "What is 17 times 3?": 51
serve praxis on 127.0.0.1:18371: up to 30 sequences of 8192 tokens, 14.68 GB planned of 14.78 GB
POST /v1/chat/completions: "51", finish_reason stop
```

`doctor praxis` lists NVFP4 as a format that fits, with a refusal and a path on the GPU server as its load (Findings).

### 13c and 14d. The image, on the GPU server

Built from `4a85ea3`'s `docker/Dockerfile` on the host network, so the fetch stage reached this round's host:

    docker build --network host -t dromos-rc:0.1.0-rc2 \
      --build-arg DROMOS_VERSION=0.1.0-rc2 --build-arg DROMOS_RELEASES=http://127.0.0.1:18370 .

```
fetch stage   dromos-linux-x86_64-cuda12.tar.gz: OK (sha256sum -c against the release's SHA256SUMS)
build         rc 0 in 1.5 minutes
image         672,162,371 bytes; source label https://github.com/lernex/dromos
config        ENTRYPOINT dromos serve --host 0.0.0.0 --port 8080, CMD praxis
--version     dromos 0.1.0-rc2 (linux-x86_64-cuda12, be70353)
```

Then the image's own ENTRYPOINT and CMD with `--gpus all` and the models folder mounted, under the GPU lock:

```
serve           Serving praxis on http://0.0.0.0:8080/v1: up to 30 sequences of 8192 tokens, 14.68 GB planned of 14.78 GB
POST /v1/chat/completions ("What is 17 times 3?", thinking off): "51", finish_reason stop
doctor          (docker exec) sm_120; CUDA driver 13.2, runtime 12.8; kernels for sm_80 sm_86 sm_89 sm_90 sm_100 sm_120;
                backend CUDA, /opt/dromos/lib/libdromos_kernels.so; missing none; tuning from the image
update --check  this dromos was installed with docker. Update it with `docker pull ghcr.io/lernex/dromos`.
stopped         docker stop; no compute apps on the GPU afterwards
```

The image stays on the GPU server as `dromos-rc:0.1.0-rc2` and is not pushed anywhere.

### 15b. `dromos update` end to end

The same method as round 2's 15a, in one container: `relhost.py` answers `/latest` with a 302 to `/tag/v0.1.0`, and a
test-only 0.1.0 x86_64 CPU tarball built from `4a85ea3` with `LX_VERSION=0.1.0` plays the newer release.

```
install 0.1.0-rc2 pinned        dromos 0.1.0-rc2 (linux-x86_64-cpu, 4a85ea3)
run 1 (no cache yet)            51, no notice; afterwards the cache reads latest=0.1.0
run 2                           51, and on stderr: dromos 0.1.0 is out (you have 0.1.0-rc2). Run `dromos update` to install it.
update                          dromos 0.1.0 (linux-x86_64-cpu, 4a85ea3); ~/.dromos/versions 0.1.0 0.1.0-rc2, current -> 0.1.0
run 3                           51, no notice
update --rollback               dromos 0.1.0-rc2 (linux-x86_64-cpu, 4a85ea3)
```

Against the plain hosts of rows 4 and 10 to 12, `update --check` printed "the release host names no latest release",
as it should with no `/latest` redirect.

### 25a. Homebrew and pip on this Mac

The formula is `packaging/homebrew-tap/Formula/dromos.rb` as `bump-formula.sh` rewrites it for this release (its
`url`, `version` and `sha256` pointed at the Mac's release host), installed from a tap made for the row with `brew
tap-new --no-git lernex/rehearsal`. Homebrew 7 refuses a formula from an untrusted tap, so the row trusted it with
`brew trust --formula` and untrusted it afterwards.

```
brew install       /opt/homebrew/Cellar/dromos/0.1.0-rc2: 12 files, 2.9MB
--version          dromos 0.1.0-rc2 (darwin-arm64, 4a85ea3)
doctor             rc 0, Metal backend, missing none
update --check     this dromos was installed with homebrew. Update it with `brew upgrade dromos`.
brew test          rc 0 (the formula's test runs --version and doctor)
brew audit         --strict: rc 0, no problems
brew style         1 file inspected, no offenses detected
run praxis         51 on Metal
uninstall, untap   rc 0; no dromos formula, trust or tap left
```

pip, from the same tarball:

```
build_wheel.py     dromos-0.1.0rc2-py3-none-macosx_14_0_arm64.whl (1.26 MB), macOS target 14.0; twine check passes
pip install        rc 0 in a fresh Python 3.12.13 venv
--version          dromos 0.1.0-rc2 (darwin-arm64, 4a85ea3)
update --check     this dromos was installed with pip. Update it with `pip install -U dromos`.
run praxis         51 on Metal
```

The first pass of rows 25 and 26 printed rc 127 for the prompt: the row script called `timeout`, which macOS does not
have. The rerun timed the prompt another way and both passed; nothing in the release changed between them.

### 27a. The aarch64 wheel (Docker Desktop on this Mac)

`build_wheel.py` from the aarch64 tarball makes `dromos-0.1.0rc2-py3-none-manylinux_2_28_aarch64.whl`. In Ubuntu
24.04 (Python 3.12.3): `twine check` passes, `pip install` rc 0, `dromos 0.1.0-rc2 (linux-aarch64-cpu, 4a85ea3)`,
`doctor` rc 0 with missing none, and `update --check` names pip.

### 28a and 29a. The cuda12 wheel

`build_wheel.py --tree` from the cuda12 tarball, as `pypi.yml` runs it, makes
`dromos-0.1.0rc2-py3-none-manylinux_2_28_x86_64.whl` (60 MB). It leaves `libcublasLt.so.12` and `libcudart.so.12` out
and requires NVIDIA's wheels instead (`nvidia-cuda-runtime-cu12>=12.8.90,<13`, `nvidia-cublas-cu12>=12.8.5.5,<13`),
so the 752 MB cuBLASLt comes from NVIDIA's wheel, not ours. `twine check` passes.

```
28a  GPU server, fresh Python 3.14 venv
     pip install    rc 0: dromos 0.1.0rc2, nvidia-cublas-cu12 12.9.2.10, nvidia-cuda-runtime-cu12 12.9.79, nvidia-cuda-nvrtc-cu12 12.9.86
     --version      dromos 0.1.0-rc2 (linux-x86_64-cuda12, be70353)
     doctor praxis  rc 0: RTX 5070 Ti, CUDA driver 13.2, runtime 12.9, backend libdromos_kernels.so from the wheel, missing none
     run praxis     51 on the GPU
     update --check this dromos was installed with pip. Update it with `pip install -U dromos`.
29a  ubuntu:24.04 container on the GPU server, no GPU passed in, Python 3.12.3
     pip install    rc 0, the same NVIDIA wheels
     --version      dromos 0.1.0-rc2 (linux-x86_64-cuda12, be70353)
     doctor         rc 0: the CPU backend (i5-14500, AVX2, FMA, F16C, AVX-VNNI, 68 GB/s measured on 14 threads),
                    libdromos_cpu.so from the wheel, missing none
     run praxis     51 on the CPU
```

pip takes the newest 12.x of NVIDIA's wheels, 12.9 here, under a build made with 12.8. CUDA's minor-version
compatibility is what lets that run, and row 28 is it running.

## Round 3: Windows x64 (2026-10-28)

The first Windows builds, from `main` at `c3101ba` (Dromos names) with `LX_ALLOW_DIRTY=1`, in the x64 Windows 11 VM on
the GPU server: QEMU/KVM with `-cpu host` on an i5-14500 (AVX2 and AVX-VNNI, no AVX-512), 6 vCPUs, 8.5 GB, no GPU.
Visual Studio Build Tools 17.14 (MSVC 14.44) with its own CMake and Ninja, Python 3.14.8, and CUDA 12.8 from
`scripts/release/cuda-toolkit.ps1` (nvcc 12.8.93, cudart 12.8.90, cuBLAS 12.8.5.5). Every zip came from
`scripts/release/build.ps1`, the script `release.yml`'s Windows jobs run, and every install from `install.ps1` under
Windows PowerShell 5.1 (`irm | iex`), against two release hosts on the server shaped like GitHub's: `/latest` and
`/latest/download/NAME` answer 302, each asset redirects once more as GitHub's do to its object store, and every
asset is `application/octet-stream` (`relhost-gh.py` in `~/installer-win`, reached from the VM as 10.0.2.2). One
says the latest release is 0.1.1, the other 0.1.2; the three CPU zips 0.1.0, 0.1.1 and 0.1.2 differ only in
`LX_VERSION`.

| # | What | Where | Result |
|---|------|-------|--------|
| 16 | `build.ps1 windows-x86_64-cpu`, clean: tree check, `dumpbin /dependents`, the tree's `--version`, the zip | the VM | pass (16a) |
| 17 | `build.ps1 windows-x86_64-cuda12`: nvcc 12.8 with `cl` as host compiler, `LX_ARCHS=all`; then on a PC without an NVIDIA driver: `--version`, `doctor`, one prompt, `install.ps1` choosing and forcing it, its CUDA DLLs missing, the cuda12 wheel with NVIDIA's wheels | the VM | pass, as far as a PC without a GPU goes (17a) |
| 18 | `install.ps1` pinned to 0.1.0: the flavor it picks, the user `PATH` in a new terminal, `doctor`, `doctor praxis`, one prompt on the CPU, a prompt outside ASCII, `serve` and one curl, the daily notice | the VM | pass (18a) |
| 19 | `dromos update` end to end: `--check`, update while the old version serves, a running version retired, `--rollback`, the sweep; `install.ps1` re-run, already current, failing under iex and as a file | the VM, both hosts | pass (18a) |
| 20 | HTTPS through WinHTTP: `update --check` against github.com, a 404 refused cleanly, `pull` of a public model from the Hugging Face hub | the VM, the internet | pass (18a) |
| 21 | A real console (ConPTY, over `ssh -tt`): UTF-8 in and out, Ctrl-C during a reply and at the prompt, Ctrl-Z to quit | the VM | pass (21a) |
| 22 | The `win_amd64` wheel: `build_wheel.py` from the CPU zip, `pip install` in a fresh venv, the launcher | the VM, Python 3.14.8 | pass (22a) |
| 23 | `install.ps1`'s choices and refusals: flavor from a stub `nvidia-smi` (driver and compute capability boundaries), a corrupt zip, a bad `DROMOS_FLAVOR`, a missing release, an unreachable host, a running version | the VM, fake builds | pass, under the old names (23a) |
| 24 | `install.ps1` under PowerShell 7: `irm` on an asset served as `application/octet-stream`, pinned and latest installs, `update` and `--rollback` from that install, a failure under iex and as a file, the user `PATH` | the VM, PowerShell 7.6.9 | pass (24a) |

The test binaries ran in the VM too: `dromos_tests` 34 of 34 cases, `lx_cpu_isa_check` 664 of 664 checks of the CPU
kernels at every level the VM has (generic, AVX2, AVX-VNNI), `lx_text_tests` 12 of 12, and `dromos_unit` 51 of 55:
the four that fail ask the CUDA device for its architecture and the `memory_fuse` kernel, and need the GPU server.

### 16a. The CPU zip

```
build.ps1: dromos.exe links WS2_32.dll, WINHTTP.dll, KERNEL32.dll
build.ps1: dromos_cpu.dll links KERNEL32.dll
build.ps1: dromos 0.1.0 (windows-x86_64-cpu, c3101ba)
build.ps1: C:\lxw\rel-0.1.0\dromos-windows-x86_64-cpu.zip 1.7 MB (2 files in bin\)
```

Nothing links the Visual C++ runtime, so a clean Windows runs it. Two runs of `pack_zip.py` over one tree give the
same bytes.

### 17a. The CUDA 12 zip, on a PC without a GPU

```
build.ps1: dromos_kernels.dll links cudart64_12.dll, cublasLt64_12.dll, KERNEL32.dll
build.ps1: dromos 0.1.2 (windows-x86_64-cuda12, c3101ba)
build.ps1: C:\lxw\rel-cuda\dromos-windows-x86_64-cuda12.zip 514.3 MB (5 files in bin\)
```

`bin\` holds `dromos.exe` (2.8 MB), `dromos_cpu.dll` (1.7 MB), `dromos_kernels.dll` (139 MB, real code for sm_80, 86,
89, 90a, 100a and 120a), `cudart64_12.dll` (0.6 MB) and `cublasLt64_12.dll` (675 MB), with NVIDIA's EULA in
`share\dromos\licenses`. The build needs `95e43a2` on `main`: before it, `gemm_skinny.cu` called the cluster API
unguarded, and every `LX_ARCHS=all` build failed on sm_80, 86 and 89, on Linux too.

The VM has no NVIDIA driver, so the CUDA library loads and finds no device:

```
--version                 dromos 0.1.2 (windows-x86_64-cuda12, c3101ba)
doctor                    backend    CPU, dromos_cpu.dll (C:\lxw\cuda-tree\bin\dromos_cpu.dll)
                          skipped    CUDA, dromos_kernels.dll: found no NVIDIA GPU (there is none, or its driver is
                                     missing or too old)
                          missing    none
run praxis --think off    "What is 17 times 3?" 51, on the CPU
install.ps1               cpu build, no NVIDIA driver found, so no CUDA (the release lists both zips)
DROMOS_FLAVOR=cuda12      cuda12 build, DROMOS_FLAVOR asks for it
                          WARNING: cuda12 needs an NVIDIA driver of 527.41 or newer, and this PC has none: the
                          engine may not start
                          installed dromos 0.1.2 in %LOCALAPPDATA%\Programs\Dromos\versions\0.1.2; doctor as above
cudart64_12.dll away      skipped    CUDA, dromos_kernels.dll: did not load: cudart64_12.dll, which it needs, is not
                                     beside it, in System32 or on PATH (error 126)
LX_BACKEND=cuda           error: cannot use device 0: CUDA driver version is insufficient for CUDA runtime version
                          (CUDA error 35)
the wheel                 build_wheel.py leaves cudart64_12.dll, cublasLt64_12.dll and the EULA to NVIDIA's wheels:
                          55.9 MB. pip resolves nvidia-cuda-runtime-cu12 12.9.79, nvidia-cublas-cu12 12.9.2.10 and the
                          nvidia-cuda-nvrtc-cu12 12.9.86 that cuBLAS 12.9 requires, 111 s in all
  through the launcher    skipped    CUDA, dromos_kernels.dll: found no NVIDIA GPU (...), loaded from NVIDIA's DLLs
  the engine directly     skipped    CUDA, dromos_kernels.dll: did not load: cudart64_12.dll and cublasLt64_12.dll,
                                     which it needs, are not beside it, in System32 or on PATH (error 126)
```

The `skipped` row is new in this round, on every platform. Before it, a cuda12 build without a GPU printed the same
`doctor` as the CPU build, with no word on why, and a CUDA library that did not load on Windows said only "The
specified module could not be found" about a file that was there. A CPU build, a Mac and a machine whose GPU the
CUDA library takes print no `skipped` row. The rows above ran a `dromos.exe` built with it, in the c3101ba tree.

The wheel names NVIDIA's wheels as floors below the next major (`>=12.8.90,<13`), so pip takes 12.9: CUDA 12's
minor-version compatibility runs a 12.8 build on them, and the zip carries real code only, no PTX to JIT. A
dependency DLL is also found on `PATH` when it is not beside the library (dlopen's fallback, there for those wheels),
so a cuda12 install's `current\bin` on the user `PATH` lends its cudart to any other copy; the missing-DLL rows above
ran with that entry taken off `PATH`.

### 18a. Install, run, serve, update (Windows PowerShell 5.1, `irm | iex`)

```
install 0.1.0 pinned        dromos: Windows x64: cpu build, no NVIDIA driver found, so no CUDA
                            dromos: installed dromos 0.1.0 in %LOCALAPPDATA%\Programs\Dromos\versions\0.1.0
                            dromos: added %LOCALAPPDATA%\Programs\Dromos\current\bin to your user PATH: open a new terminal to use dromos
new terminal                C:\Users\lernex\AppData\Local\Programs\Dromos\current\bin\dromos.exe, dromos 0.1.0 (windows-x86_64-cpu, c3101ba)
doctor                      backend CPU, dromos_cpu.dll (...\Dromos\versions\0.1.0\bin\dromos_cpu.dll), missing none
run praxis --think off      "What is 17 times 3?" 51, 12 tok/s, 18 s with the load (BF16 Praxis, 6.2 GB)
run, outside ASCII          "Wie heißt die Hauptstadt von Österreich?" Wien., stdout valid UTF-8; the daily notice on
                            stderr: dromos 0.1.1 is out (you have 0.1.0). Run `dromos update` to install it.
serve praxis + curl         up in 2 s; "Name the largest planet" Jupiter in 2 s; /v1/models lists praxis
update --check              dromos 0.1.1 is available (you have 0.1.0).
update (0.1.0 serving)      dromos 0.1.1 is installed and current. dromos 0.1.0 is kept
update (host says 0.1.2)    dromos 0.1.2 is installed and current. dromos 0.1.1 is kept
                            versions: .retired-0.1.0-10072 0.1.1 0.1.2 (0.1.0 renamed aside, its serve still answering /health)
update --rollback           dromos switched back to 0.1.1. The next run uses it.
update again                versions: 0.1.1 0.1.2 (the leftover swept once the serve stopped)
install.ps1 (latest)        installed dromos 0.1.2; your user PATH already has ...\current\bin
install.ps1 again           dromos 0.1.2 is already installed and current
user PATH                   one entry, %LOCALAPPDATA%\Programs\Dromos\current\bin, REG_EXPAND_SZ
DROMOS_VERSION=9.9.9        under iex: dromos: there is no release 9.9.9 at ... (check DROMOS_VERSION), the session stays;
                            powershell -File install.ps1: the same line, exit 1
github.com/cli/cli          update --check: dromos 2.103.3 is available (TLS, the /latest redirect); update: GET
                            .../SHA256SUMS answered HTTP 404, exit 1, nothing changed
pull HuggingFaceTB/SmolLM2-135M   269 MB over HTTPS with the hub's CDN redirects, 85 s, dromos.json written
```

`current` and `previous` are junctions, and `dromos update` retargets `current` in place. Windows lets the folder
of a running program be renamed, so the version a serve was running from moved aside whole and kept serving;
`install.ps1` asks the process list instead and keeps such a version until it stops.

### 21a. A real console

`dromos run praxis --think off` in a ConPTY over `ssh -tt`, typed input:

```
> Translate 'thank you' into Japanese. Reply with only the Japanese words.
ありがとう (arigatou)
> Write a 600-word story about a lighthouse keeper.
The beam of light swept across ... the wind’s mournful song ... preferring to dwell in [stopped]
153 tokens, 21 tok/s
> half-typed line        (Ctrl-C at the prompt: the line is dropped and the prompt asks again)
> Was ist 2 plus 2? Antworte mit einem Wort auf Deutsch.
> ^Z                     (Ctrl-Z, Enter: the session ends)
```

### 22a. The wheel

`build_wheel.py --plat win_amd64` from the 0.1.2 CPU zip ("2 Windows images are PE32+ AMD64"), then in a fresh venv:
`pip install` ok, `dromos --version` and `python -m dromos --version` print `dromos 0.1.2 (windows-x86_64-cpu,
c3101ba)`, `doctor` loads `dromos_cpu.dll` from site-packages, the channel is `pip` and `dromos update` says `pip
install -U dromos`, and an unknown command exits 2 through the launcher. NVIDIA's `win_amd64` wheels at the floors the
wheel names carry `nvidia/cuda_runtime/bin/cudart64_12.dll` and `nvidia/cublas/bin/cublasLt64_12.dll` (read from their
zip directories on PyPI), the folders the launcher puts on `PATH`.

### 23a. `install.ps1`'s own matrix

Run before the rename with fake `lernex.exe` builds, a stub `nvidia-smi` and a local server: RTX 4070 on driver
560.94 gets cuda12; 516.94 (below 527.41) gets cpu, as do a GTX 1650 (sm_75), no `nvidia-smi`, an unreadable
compute capability and a release without a cuda12 zip; 527.41 and sm_80 exactly get cuda12. A zip cut in half fails
the checksum and changes nothing; a bad flavor, a missing version and an unreachable host each print one line. A
version running from `versions\X` is never replaced: Windows allowed the rename that was meant to catch it, so the
installer checks the process list first. ARM64 and builds below 17763 are refused (simulated).

### 24a. PowerShell 7

PowerShell 7.6.9, portable from PowerShell's own release (its SHA-256 matched the digest GitHub lists for the
asset), against the same GitHub-shaped host in the VM with 0.1.1 and 0.1.2:

```
irm .../install.ps1       System.String, 21014 chars (PowerShell 7 decodes an octet-stream body as text, as 5.1 does)
DROMOS_VERSION=0.1.1      installed dromos 0.1.1 ... (was 0.1.2): the cuda12 0.1.2 from 17a, now previous
latest                    installed dromos 0.1.2 ... (was 0.1.1): the cpu zip in place of the cuda12 one, same version
update --rollback         dromos switched back to 0.1.1
update                    dromos 0.1.2 is installed and current. dromos 0.1.1 is kept
DROMOS_VERSION=9.9.9      under iex: there is no release 9.9.9 ..., and the session goes on; pwsh -File: exit 1
user PATH                 one entry, %LOCALAPPDATA%\Programs\Dromos\current\bin, ExpandString
```

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

- Fixed in round 4 (`4a85ea3`): the rename (`db23c4e`) wrote the release address as `github.com/dromos/dromos` into
  `install.sh`, `engine/core/update.cpp`, `docker/Dockerfile` and the Homebrew formula and its bump script. The
  repository is `lernex/dromos`, so every install, update and image build would have asked a host that does not
  exist. They now name `github.com/lernex/dromos`, and `release.yml`'s `release-address` job stops a release whose
  `install.sh`, `update.cpp` or `Dockerfile` names any address but its own repository's.
- Fixed in round 4 (`6822467`): the CUDA flavors' toolkit came from NVIDIA's package repositories, whose RHEL indexes
  were broken for hours on October 29, and which serve the newest packages of a series. It now comes from NVIDIA's
  redistributable archives, pinned by manifest and sha256 (How it was built, round 4). The Windows flavor's
  `cuda-toolkit.ps1` pins the same manifest.
- Fixed after round 4 (`d290c4d`): `doctor praxis` on a BF16 checkpoint listed NVFP4 under the formats that fit,
  with a refusal as its load, and the refusal named a GPTQ checkpoint by its folder on the GPU server
  (`/home/aiadmin/kronos-scratch/...`); `run`'s refusal listed it the same way. NVFP4's load is now the command that
  runs it, `dromos run Lernex/Metis-1.6-Praxis-NVFP4` (or Logos), on a line of its own; "fits" lists only what this
  checkpoint loads; and `release.yml` fails a release whose engine, CLI or installers hold a `/home/` or
  `kronos-scratch` path. The GPTQ folder behind that name loads in rc2 as stored NVFP4 (`doctor` on it).

- Windows lets a running program's folder be renamed. `dromos update` uses that: a version still serving is moved
  aside whole and keeps running, and the next update deletes what is left (18a). `install.ps1` instead asks the
  process list before it replaces or removes a version, since a rename cannot tell it one is running (23a).
- `stdout` redirected to a file on Windows ends lines with CRLF (the C runtime's text mode), as Windows programs do.
  The console and the server are unaffected.
- The CUDA toolkit's network installer exits 0xE0E00019 when it runs without a desktop session, as it would on a
  runner. `cuda-toolkit.ps1` takes NVIDIA's redistributable archives instead, checked against NVIDIA's manifest.
- The Windows binaries are not signed. `install.ps1` downloads with `Invoke-WebRequest` and unpacks with `tar.exe`,
  which leave no Mark of the Web, so nothing warns. A zip downloaded in a browser carries the mark, and SmartScreen
  warns on the first run of its unsigned `dromos.exe`, as Gatekeeper does for the darwin tarball (below).

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

- `Lernex/Metis-1.6-Praxis-NVFP4` and `Lernex/Metis-1.6-Logos-NVFP4`, the names NVFP4's line gives (`d290c4d`), must
  be public on launch day, or the two lines of `kNvfp4` in `engine/core/checkpoint.cpp` emptied before the freeze so
  the line says none is published. The owner's decision. Row 10's `doctor praxis` on the freeze build shows which.
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
- Windows needs an NVIDIA PC for what the VM cannot run (round 3): the cuda12 flavor loading `dromos_kernels.dll`
  against a real driver, its kernels, cuBLASLt and NVML (`nvml.dll`) on Windows, and its speed against Linux;
  `install.ps1` choosing cuda12 from a real `nvidia-smi`; the `win_amd64` wheel with NVIDIA's wheels on a GPU. A GPU
  that drives a display is under Windows's TDR: a kernel running over 2 seconds is reset. Decode steps are far
  shorter; the longest prefill chunk is the one to time there.
- `release.yml`'s Windows jobs have not run on GitHub (Actions is held until November 1). Round 3 ran their script,
  `build.ps1`, by hand in the VM. `install.ps1` ran under Windows PowerShell 5.1 and PowerShell 7.6.9 (row 24).
- On the CPU backend, a model larger than the free RAM is planned as a GPU's would be: Praxis BF16 (5.89 GB) on the
  8.5 GB VM offloads its experts "cached in 5.85 GB of VRAM, all of them in pinned RAM", in `run` and `doctor` alike,
  on every platform. Both are the same RAM there. With the engine session (`plan.cpp`, `offload_plan.cpp`).
- Windows on Arm has no build, and `install.ps1` refuses it. The x64 CPU zip under Windows's x64 emulation is
  untested.
