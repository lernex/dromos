# The HIP backend

`engine/hip` is the kernel library's C ABI (`engine/cuda/kernels.h`) on AMD GPUs: RDNA 3 and 4
(wave32, gfx1100 and gfx1201) and CDNA 3 (wave64, gfx942). It is ported from `engine/cuda`, which
stays the source of truth. Nothing under `engine/cuda` or `engine/model` is changed by the port.

## Status

| Check | Result |
| --- | --- |
| AMD compile, ROCm SDK 7.9rc `amdclang++`, `--offload-arch` per target | gfx942, gfx1100, gfx1201: 0 errors, 0 warnings each |
| NVIDIA test shim (RTX 5070 Ti, sm_120a), `tests/kernels` | 3743 passed, 42 skipped (the CUDA library: 3784 passed, 1 skipped) |
| `dromos doctor` with `LX_BACKEND=hip` on the shim | missing none |
| End to end, Praxis BF16, HIP library on the NVIDIA shim | prefill bitwise equal to the CUDA library; decode within bf16 noise (below) |
| Run on AMD hardware | not done: no AMD GPU was available to this port |

Only the compile results and the NVIDIA runs are measured. Nothing in this directory has executed
on an AMD GPU. The wave64 path (gfx942) is compiled but never run.

## Layout

- `translate.py`: generates `src/` from `engine/cuda`. Subcommands `generate`, `save FILE`, `check`.
- `src/`: generated. Edit it only to make a structural change, then `save` it (see the recipe).
- `patches/`: one patch per file that needs more than the rules (warp size, cluster-free paths).
- `include/lx_hip.h`: the portability layer every translated source includes.
- `nvidia/`: HIP names over the CUDA runtime, for the NVIDIA test build only.
- `CMakeLists.txt`: `LX_HIP_PLATFORM=amd` (default) or `nvidia`.
- `e2e/`: the dromos CLI and the Metis model on this library, with `prepare.sh`.
- `verify/`: differential checks, CUDA library against HIP library on the same inputs (NVIDIA only).
- `bench/bench_kernels.py`: times the decode kernels of any library with CUDA/HIP events.
- `amd/run_amd.sh`: one command on a rented AMD host: build, `tests/kernels`, benchmark.

## Translation recipe

1. Change the CUDA source in `engine/cuda/`.
2. `python3 engine/hip/translate.py generate` rebuilds `src/` from the rules and the patches. If a
   patch no longer applies, it stops and names the file: re-port that change by hand in `src/`.
3. For a structural port change, edit `src/FILE` by hand, then
   `python3 engine/hip/translate.py save FILE`. That writes `patches/FILE.patch`.
4. `python3 engine/hip/translate.py check` must pass before committing.

The rules are mechanical: `cuda*` becomes `hip*`, `cublasLt` becomes `hipblasLt`, `__nv_bfloat16`
becomes `__hip_bfloat16`, and the NVIDIA-only constructs map to the portable forms in `lx_hip.h`.
`kernels.h` and `tunable.h` are copied verbatim, because they are the ABI.

## Portability decisions

- **Wave size is a target property.** `LX_WARP` is `__AMDGCN_WAVEFRONT_SIZE__` on the device (32 or
  64). Host code asks `lx_warp_size()`, which queries `hipDeviceAttributeWarpSize`. Block sizes are
  whole wavefronts, and the shuffle mask is `LX_FULL_MASK`, 64 bits on wave64. No code assumes 32.
- **`lx_arch` returns the gfx family**: gfx1100 gives 110, gfx1201 gives 120, gfx942 gives 94.
- **Thread-block clusters are removed.** The CUDA library splits decode work across an 8-block
  cluster in three places: the attention split mode (few items), the memory fuse, and the mhc read.
  The port runs one block per item or token instead. The results are bitwise equal in most cases
  (below), but the decode kernels are slower on NVIDIA, which is measured below.
- **`cp.async` is a synchronous copy** (`lx_pipeline_memcpy_async`). Correctness rests on the
  existing `__syncthreads` ordering. The copy no longer overlaps the math.
- **Programmatic dependent launch** is a no-op. The stream orders the kernels.
- **Cooperative launch** (the decode program's persistent kernel, whose tasks wait on other blocks' tasks, so every
  block must be resident) goes through `hipLaunchCooperativeKernel` with the arguments packed, on both platforms
  (`lx_launch_kernel_ex` in `include/lx_hip.h`). The NVIDIA shim takes the same path.
- **The opt-in shared-memory limit** (`cudaDevAttrMaxSharedMemoryPerBlockOptin`) is AMD's per-block LDS limit.
- **No K split.** The quantised GEMM's K split runs on thread-block clusters, which HIP does not have, so `split_for`
  (`patches/gemm_w.cu.patch`) never splits on HIP. The split variants would answer `hipErrorNotSupported`.
- **hipBLASLt alignment.** hipBLASLt 7.2.4 has no minimum-alignment preference (the CUDA source sets
  four). The NVIDIA shim maps the names, so CUDA keeps them. On AMD the heuristic is not told the
  operands' alignment, so `gemm.cu` refuses an operand narrower than 16 bytes with
  `HIPBLAS_STATUS_NOT_SUPPORTED`. The 16-byte bound is a guess. Check it on hardware before trusting it.

## The NVIDIA test shim

The shim makes the same sources compile with `nvcc` and run the same tests, so the AMD port can be
checked against the CUDA library on one GPU. `LX_NVIDIA_SHIM` is defined only there.

Why not HIP's own NVIDIA backend: the ROCm `nvidia_detail` headers that carry the CUDA backend exist
only in ROCm clr 5.7.x. Against CUDA 13.4 they fail (`cudaDeviceProp` fields removed,
`cudaMemPrefetchAsync` signature changed). The shim is a thin set of HIP names over the CUDA runtime
instead, and it is never part of the AMD build.

## Toolchain used

- Compiler: `amdclang++` from the ROCm SDK wheel 7.9rc (`_rocm_sdk_core`), with its HIP headers and
  device bitcode. The CMake build uses the C++ language with `-x hip --offload-arch=...`, because the
  wheel carries none of the ROCm CMake packages that CMake's HIP language needs.
- hipBLASLt headers: ROCm 7.2.4 `library/include`, with the export and version headers the public
  header includes supplied as stubs on the build server.
- Rejected: Ubuntu clang 19 with ROCm 6.4.4 headers. It fails inside `__clang_cuda_complex_builtins.h`.

## Measured results

### Kernel level, CUDA library against HIP library (NVIDIA), same inputs

`verify/diff_decode.py`, `verify/diff_attn.py` and `verify/diff_gemv.py` bind both libraries and
compare outputs element by element.

- `memory_fuse`, decode (T up to 8): bitwise on the summary. The streams differ by at most one float32
  ulp (4.8e-7 at magnitude 4.6), on 6 to 20 percent of the elements, by shape. This is the cluster reduction in the
  gate dot, now summed in one block. Prefill-size shapes are bitwise.
- `mhc_read`, `memory_attend`: bitwise at every shape tested, decode and prefill.
- `attention_varlen`: bitwise at decode (1 and 8 sequences, positions 0 to 4000, the split path
  included) and at prefill, for the output and both caches.
- `gemv` int8 (five N by K shapes, M 1 to 8), `quantize_w8`, `moe_experts_w8` (T 1 to 300): bitwise.

### End to end, Praxis BF16, `dromos eval` on both builds, scored with `tools/compare.py`

Top-20 KL against the native FP32 reference, mean per position.

| Prompt (tokens) | CUDA prefill | HIP prefill | CUDA, 8-token prefill then decode | HIP, 8-token prefill then decode |
| --- | --- | --- | --- | --- |
| chat_think (16) | 0.00515 | 0.00515 | 0.00419 | 0.00345 |
| chat_direct_tools (176) | 0.00977 | 0.00977 | 0.01439 | 0.01423 |
| prose (300) | 0.00129 | 0.00129 | 0.00246 | 0.00297 |

- Prefill: HIP logits are bitwise equal to the CUDA logits on all three prompts.
- Decode: HIP against CUDA, top-20 KL mean 0.00058, 0.0105 and 0.0011 (max 0.0064, 0.57 and 0.098),
  top-1 agreement 100%, 97.7% and 99.3%. The decode gap is the same size as each engine's own error
  against the reference. The best explanation is the one-ulp fuse difference above, carried through
  bf16 storage across layers. That is inferred, not isolated: an end-to-end ablation would need the
  CUDA cluster code back.

### Cost of removing the clusters (NVIDIA, median microseconds, back to back)

The GPU was shared with other sessions, so read these as relative numbers only.

| Kernel | CUDA library | HIP library (shim) |
| --- | --- | --- |
| gemv int8 N6144 K2048 M1 / M8 | 13.7 / 27.3 | 13.8 / 27.2 |
| moe int8 T1 / T8 | 14.8 / 40.4 | 16.0 / 46.6 |
| attention decode, 1 sequence @3000 | 30.2 | 105.2 |
| attention decode, 8 sequences @3000 | 162.7 | 159.9 |
| memory fuse T8 S4 D2048 | 7.4 | 15.0 |
| mhc read T8 S4 D2560 | 7.6 | 11.0 |
| memory attend T8 S4 M256 E8 | 7.1 | 6.9 |

The gemv rows come from the first run and the rest from the second; both were taken in one session.
The single-sequence attention is the largest cost. It is the case the CUDA split mode exists for.

## Needs AMD hardware

1. Run `tests/kernels` on gfx1100, gfx1201 and gfx942. gfx942 is the only wave64 target, so its path
   has never executed.
2. Decode attention with few items: restore the split (two kernels, a partial pass and a merge)
   instead of one block per item. This is the 3.5x NVIDIA gap above, and AMD's answer may differ.
3. Decode fuse and mhc read: one block per token gives little parallelism. Tune the shape on AMD.
4. Tile sizes. Attention's `kBlockN` is 32 here against 64 in CUDA, and `kStages` is 3. Both were
   chosen for the 64 KB LDS budget at Dh = 128 (3 stages of K and V is 48 KB). Confirm the LDS limit
   per workgroup on each target, and tune.
5. Asynchronous copies: replace the synchronous staging with the target's async global-to-LDS path.
6. Tensor cores. There is no WMMA (RDNA 3 and 4) or MFMA (CDNA 3) path yet. The CUDA source has none
   to port; the dense products go through hipBLASLt.
7. hipBLASLt on AMD: its heuristic, and the 16-byte alignment guard in `gemm.cu`.
   The GEMM is not pinned: HIP prefill rows are not batch-invariant (CUDA pins its algorithm per shape, so a row's
   bits do not depend on the step around it). Pinning needs the hipBLASLt algorithm-config names (algorithm
   config get and check, split-K, reduction scheme), which are to be checked against the ROCm headers.
8. Graph capture and replay on HIP. The model replays decode steps as graphs.
9. `lx_arch` and `hipDeviceAttributeWarpSize` on each board.
10. The model on AMD: apply the handoff diff and run the Praxis KL there.

## The model handoff

`docs/handoff/hip-metis.diff` ports `engine/core/device.*`, `engine/model/metis.*`, `engine/model/glue.*`
and `cli/eval.cpp`, `cli/inspect.cpp` to `lx_rt_*`, so the model calls the library's runtime section
instead of the CUDA runtime. The model code then has no GPU vendor headers, except `glue.cu` (the
four small kernels, now HIP names). The diff is not applied to `engine/model`. `e2e/prepare.sh` applies
it to a copy.

`inspect` loses its device-memory lines in the port. `lx_rt_*` has no memory query. Adding one means
an ABI change in `engine/cuda`, which this port does not make.

## Running it

NVIDIA test build and kernel tests, from the repository root (through the slot helper):

    scripts/remote.sh hip run 'cmake -S engine/hip -B build/nv -DLX_HIP_PLATFORM=nvidia -DLX_HIP_CUDA_ARCHS=120a && cmake --build build/nv -j 4'
    scripts/remote.sh hip x 'LX_KERNELS=$PWD/build/nv/libdromos_hip.so $PY -m pytest -q tests/kernels'

End to end on the NVIDIA shim:

    engine/hip/e2e/prepare.sh build/e2e-tree
    cmake -S engine/hip/e2e -B build/e2e -DLX_E2E_TREE=$PWD/build/e2e-tree && cmake --build build/e2e
    build/e2e/dromos eval MODEL_DIR PROMPTS_JSON OUT_DIR [N]

AMD, on a rented host with ROCm and a ROCm build of torch:

    PY=/path/to/python engine/hip/amd/run_amd.sh /opt/rocm

It writes `out/amd/`: the build log, the test log and the benchmark. The script has not run on AMD
hardware yet.
