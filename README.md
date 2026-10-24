# Lernex engine

The inference engine for Lernex's Metis models, written from scratch for them.
Private until launch. Architecture: `docs/ARCHITECTURE.md`. Kernel ABI: `engine/cuda/kernels.h`.

```bash
scripts/remote.sh <slot> test        # build on the GPU server and run the kernel tests
scripts/remote.sh <slot> run ./build/lernex_tests   # the C++ unit tests (policy, hub, digests)
scripts/remote.sh <slot> x 'LX_BIN=build/lernex LX_MODEL=DIR $PY -m pytest -q tests/server'   # the server, end to end
```

Commands: `lernex run MODEL` (chat), `lernex serve MODEL` (OpenAI API), `lernex pull MODEL`,
`lernex list`, `lernex bench MODEL --streams 1,8,32,64`. MODEL is `praxis`, `logos`, an alias with
a format (`praxis:fp8`), a folder, or `owner/name` on the hub.
