# Lernex engine

The inference engine for Lernex's Metis models, written from scratch for them.
Private until launch. Architecture: `docs/ARCHITECTURE.md`. Kernel ABI: `engine/cuda/kernels.h`.

```bash
scripts/remote.sh <slot> test        # build on the GPU server; the kernel tests and the C++ unit tests
scripts/remote.sh <slot> run ./build/lernex_tests   # the C++ tests (scheduler, digests, hub)
LX_ARCHS=all scripts/remote.sh <slot> build   # every architecture: RTX 30 / A100 up to RTX 50, H100, B200
scripts/remote.sh <slot> x 'LX_BIN=build/lernex LX_MODEL=DIR $PY -m pytest -q tests/server'   # the server, end to end
scripts/remote.sh <slot> x 'build/lernex doctor MODEL_DIR'   # the GPU, and the plan for a model
scripts/remote.sh <slot> x 'build/lernex tune MODEL_DIR --out PATH'   # time the kernel variants on this GPU
```

Commands: `lernex run MODEL` (chat), `lernex serve MODEL` (OpenAI API), `lernex pull MODEL`,
`lernex list`, `lernex bench MODEL --streams 1,8,32,64`, `lernex update` (the latest release, for installs made from a
release tarball), `lernex version`. MODEL is `praxis`, `logos`, an alias with
a format (`praxis:fp8`), a folder, or `owner/name` on the hub.

Weights run in the format the file stores: no hidden quantization, so every speed is the speed of a named
format. `--weights bf16|int8|fp8|nvfp4|bf16x` (run, serve, bench, eval, spec, doctor) names another: INT8 and FP8
round-to-nearest from a wider file, BF16X the lossless encoding of a BF16 one, NVFP4 only from the GPTQ NVFP4
folder, nothing wider than the file. A BF16 model decodes from lossless BF16X copies when they fit (`--no-bf16x`
turns them off), and a model that does not fit the GPU runs with its routed experts offloaded rather than in a
narrower format. `lernex doctor MODEL` shows every format, what fits, and the decode ceiling of each.

`LX_ARCH_OVERRIDE=80|86|89|90` (or `--lx-arch=N` on the kernel tests) makes a newer card run the code paths
of an older architecture, so the fallbacks are tested on one GPU.
