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
`lernex list`, `lernex bench MODEL --streams 1,8,32,64`. MODEL is `praxis`, `logos`, an alias with
a format (`praxis:fp8`), a folder, or `owner/name` on the hub.

`LX_ARCH_OVERRIDE=80|86|89|90` (or `--lx-arch=N` on the kernel tests) makes a newer card run the code paths
of an older architecture, so the fallbacks are tested on one GPU.
