# Dromos

The inference engine for Lernex's Metis models, written from scratch for them.
Models: [Metis-1.6 on Hugging Face](https://huggingface.co/collections/Lernex/metis-16-6adc52e150d985721d894f6b).
Architecture: `docs/ARCHITECTURE.md`. Kernel ABI: `engine/cuda/kernels.h`.

## Install

```bash
curl -fsSL https://github.com/lernex/dromos/releases/latest/download/install.sh | sh   # Linux, macOS
```

```powershell
irm https://github.com/lernex/dromos/releases/latest/download/install.ps1 | iex           # Windows 10 1809+ x64
```

Either installer picks the build for the machine (CUDA 12 on an NVIDIA GPU with a new enough driver, Metal on Apple
silicon, the CPU otherwise), checks it against the release's `SHA256SUMS`, and puts `dromos` on your `PATH`; run it
again to upgrade, or `dromos update`. Also `pip install dromos`, `brew trust --formula lernex/tap/dromos && brew install
lernex/tap/dromos`, and `docker run --gpus all -p 8080:8080 ghcr.io/lernex/dromos praxis`. Where each puts things,
and the flavors: `docs/RELEASE.md`. Then:

```bash
dromos doctor          # the backend this machine runs and what it can do
dromos run praxis      # chat in the terminal (downloads the model the first time)
dromos serve logos     # the OpenAI API on http://127.0.0.1:8080/v1
```

## Build from source

```bash
cmake -B build -DLX_ARCHS=all -DLX_DECODE_PROGRAM=OFF && cmake --build build -j
```

CMake 3.28 or newer, a C++20 compiler, and the CUDA toolkit 12.8 or newer (the release is built with 12.8). `LX_ARCHS`
names the GPUs the build runs on: `all` is RTX 30 / A100 up to RTX 50, H100 and B200; the default, `120a-real`, is
RTX 50 only and compiles fastest. `-DLX_DECODE_PROGRAM=OFF` leaves out the opt-in persistent decode program, as the
release does: its kernel takes about 30 minutes per architecture to compile, and more than 14 GB of memory for sm_80. Without an
NVIDIA GPU, `-DLX_BACKEND_CUDA=OFF` builds the CPU backend only; on macOS the default is the Metal backend. The binary
is `build/dromos`.

## Develop

```bash
scripts/remote.sh <slot> test        # build on the GPU server; the kernel tests and the C++ unit tests
scripts/remote.sh <slot> run ./build/dromos_tests   # the C++ tests (scheduler, digests, hub)
LX_ARCHS=all scripts/remote.sh <slot> build   # every architecture: RTX 30 / A100 up to RTX 50, H100, B200
scripts/remote.sh <slot> x 'LX_BIN=build/dromos LX_MODEL=DIR $PY -m pytest -q tests/server'   # the server, end to end
scripts/remote.sh <slot> x 'build/dromos doctor MODEL_DIR'   # the GPU, and the plan for a model
scripts/remote.sh <slot> x 'build/dromos tune MODEL_DIR --out PATH'   # time the kernel variants on this GPU
```

Commands: `dromos run MODEL` (chat), `dromos serve MODEL` (OpenAI API), `dromos pull MODEL`,
`dromos list`, `dromos bench MODEL --streams 1,8,32,64`, `dromos update` (the latest release, for installs made from a
release tarball), `dromos version`. MODEL is `praxis`, `logos`, an alias with
a format (`praxis:fp8`), a folder, or `owner/name` on the hub.

Weights run in the format the file stores: no hidden quantization, so every speed is the speed of a named
format. `--weights bf16|int8|fp8|nvfp4|bf16x` (run, serve, bench, eval, spec, doctor) names another: INT8 and FP8
round-to-nearest from a wider file, BF16X the lossless encoding of a BF16 one, NVFP4 only from the GPTQ NVFP4
folder, nothing wider than the file. A model decodes its BF16 tensors (every one of a BF16 model, the wide ones of a
quantized model) from lossless BF16X copies when they fit (`--no-bf16x` turns them off), and a model
that does not fit the GPU runs with its routed experts offloaded rather than in a narrower format. `dromos doctor MODEL` shows every format, what fits, and the decode ceiling of each.

Batched steps pick each matrix product's algorithm for the rows in the step, so a request's tokens can depend on what
else is decoding beside it, as in other engines. `--batch-invariant` (run, serve, bench, spec) runs every row count of a
product on one algorithm instead: a request gets the same bits whatever shares its steps, about 15% slower at
32 streams. `dromos eval` always runs this way. Speculation never changes the output in either mode.

`LX_ARCH_OVERRIDE=80|86|89|90` (or `--lx-arch=N` on the kernel tests) makes a newer card run the code paths
of an older architecture, so the fallbacks are tested on one GPU.
