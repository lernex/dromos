# HIP backend: parked for 0.1.1

The HIP backend is not in v0.1.0's release builds. release.yml builds Linux CPU,
Linux CUDA 12, Darwin Metal, and Windows CPU and CUDA. The HIP port stays out of
0.1.0 and returns in 0.1.1.

## What blocks it

Regenerating the HIP tree from the current CUDA sources and compiling it fails in
two places.

1. The decode program is hand-ported and stale. `engine/hip/src/decode_kernel.cu`
   and `decode_sync.cuh` are the HIP copies from afee283, and they match origin's
   main byte for byte. The CUDA decode program has since moved to per-format kernel
   families and PTX device code. On gfx1100 the HIP build fails with 21 errors:
   inline-asm constraints (`=l`, `l`, `+f`), `__ldcg` overloads, and constexpr
   arguments that are not constant expressions. The HIP CMake has no
   `LX_DECODE_PROGRAM` gate, so the release flag does not remove these files.

2. The split-K choice in `engine/cuda/gemm.cu` is translated to hipBLASLt names
   that the toolchains here do not define: `HIPBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK`,
   `HIPBLASLT_ALGO_CONFIG_REDUCTION_SCHEME`, `HIPBLASLT_REDUCTION_SCHEME_NONE`, and
   `hipblasLtMatmulAlgoCheck`. The NVIDIA shim's hipBLASLt headers lack them (8
   errors in `gemm.cu`), and so do the ROCm SDK and compat headers on the AMD box.
   The earlier MIN_ALIGNMENT attributes were already guarded for the same reason.

## Options for 0.1.1

A. Guard the reduction block the way MIN_ALIGNMENT is guarded. It compiles where
   the names exist and falls back to the default algorithm choice elsewhere. This
   is cheap, but split-K selection then differs from CUDA on those toolchains, so
   HIP results are not bit-identical to CUDA until the names exist.

B. Make split-K selection independent of the hipBLASLt reduction attributes:
   choose it from our own tile table with the same rule CUDA uses. More work, and
   it keeps selection identical everywhere.

C. Upgrade the toolchain to a hipBLASLt that ships the reduction-scheme API, so the
   translation compiles unchanged. This depends on the SDK upgrade.

All three need the decode-program port in item 1 before AMD compiles.

## Where the regen lives

The regenerated tree is commit 0ed6172 on the logos worker's clone, kept as a patch
(`parked/0001` in that worker's scratch). Re-run `python3 engine/hip/translate.py
generate` after the patches are re-ported, then `check`.

## Hand-written HIP compositions to carry over

Folds A, B and C (the add_two_rows, concat4_f32 and stream_concat4 compositions in engine/hip/src/glue.cu and
engine/hip/src/elementwise.cu, with their kernels.h declarations) and the mhc write_read fusion (lx_mhc_write_read in
engine/hip/src/streams.cu, composed from lx_mhc_write and lx_mhc_read) are written by hand in src/ with no patches, so
`translate.py generate` would overwrite them. Port each into a patch when the backend is regenerated.
