# Lernex engine

The inference engine for Lernex's Metis models, written from scratch for them.
Private until launch. Architecture: `docs/ARCHITECTURE.md`. Kernel ABI: `engine/cuda/kernels.h`.

```bash
scripts/remote.sh <slot> test        # build on the GPU server; the kernel tests and the C++ unit tests
LX_ARCHS=all scripts/remote.sh <slot> build   # every architecture: RTX 30 / A100 up to RTX 50, H100, B200
scripts/remote.sh <slot> x 'build/lernex doctor MODEL_DIR'   # the GPU, and the plan for a model
scripts/remote.sh <slot> x 'build/lernex tune MODEL_DIR --out PATH'   # time the kernel variants on this GPU
```

`LX_ARCH_OVERRIDE=80|86|89|90` (or `--lx-arch=N` on the kernel tests) makes a newer card run the code paths
of an older architecture, so the fallbacks are tested on one GPU.
