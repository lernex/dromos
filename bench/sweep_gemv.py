"""Times every lx_gemv_w variant for each format, Praxis decode shape and M, against the heuristic.

The autotuner's job, done offline: lx_tune_set fixes one variant for a (kernel, M, N, K) key, the same
graph timing as bench_gemv.py measures it, and the best is reported beside the heuristic's time. One CSV
line per measurement goes to SWEEP_CSV (default: scratchpad path given by the caller) so the heuristic can
be read off the table.

Environment: LX_KERNELS, FORMATS (comma list), M (comma list), SHAPES (N:K pairs, default the Praxis set),
SWEEP_CSV (output file, default sweep.csv in the current directory).
Run: scripts/remote.sh formats x '$PY bench/sweep_gemv.py'
"""

import ctypes
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_gemv as B  # noqa: E402

lib = B.lib
lib.lx_tune_set.restype = None
lib.lx_tune_set.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.lx_tune_clear.restype = None
lib.lx_tune_clear.argtypes = []
lib.lx_tune_variants.restype = ctypes.c_int
lib.lx_tune_variants.argtypes = [ctypes.c_char_p]
lib.lx_tune_variant_name.restype = ctypes.c_char_p
lib.lx_tune_variant_name.argtypes = [ctypes.c_char_p, ctypes.c_int]

F32, BF16 = 0, 1


def time_us(copies, x, y, N, K, M, x_dtype):
    launches = []
    for lw, _ in copies:
        pointer = ctypes.pointer(lw)
        launches.append(
            lambda p=pointer: B.ok(
                lib.lx_gemv_w(
                    x.data_ptr(), x_dtype, K, p, None, F32, y.data_ptr(), F32, N, M, N, K, 1.0, 0.0, B.stream(),
                )
            )
        )
    return B.graph_us_per_launch(launches)


def main():
    formats = os.environ.get("FORMATS", "bf16,int8,fp8,nvfp4").split(",")
    ms = [int(m) for m in os.environ.get("M", "1,2,4,8").split(",")]
    if "SHAPES" in os.environ:
        shapes = [tuple(int(v) for v in s.split(":")) for s in os.environ["SHAPES"].split(",")]
    else:
        shapes = B.SHAPES
    out_path = os.environ.get("SWEEP_CSV", "sweep.csv")
    print(f"device: {torch.cuda.get_device_name(0)}; DRAM {B.DRAM_GBPS:.0f} GB/s")
    rows = []
    with open(out_path, "w") as csv:
        csv.write("format,N,K,M,variant,us,gbs\n")
        for fmt in formats:
            kernel = f"gemv_w_{fmt}"
            nvar = lib.lx_tune_variants(kernel.encode())
            names = [lib.lx_tune_variant_name(kernel.encode(), v).decode() for v in range(nvar)]
            for N, K in shapes:
                nbytes = B.weight_bytes(N, K, fmt)
                copies = [B.make_copy(N, K, fmt) for _ in range(B.copies_for(nbytes))]
                for M in ms:
                    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
                    y = torch.empty(M, N, dtype=torch.float32, device="cuda")
                    lib.lx_tune_clear()
                    heur = time_us(copies, x, y, N, K, M, BF16)
                    csv.write(f"{fmt},{N},{K},{M},heuristic,{heur:.3f},{nbytes / (heur * 1e-6) / 1e9:.1f}\n")
                    best = (heur, "heuristic")
                    for v in range(nvar):
                        lib.lx_tune_clear()
                        lib.lx_tune_set(kernel.encode(), M, N, K, v)
                        us = time_us(copies, x, y, N, K, M, BF16)
                        gbs = nbytes / (us * 1e-6) / 1e9
                        csv.write(f"{fmt},{N},{K},{M},{names[v]},{us:.3f},{gbs:.1f}\n")
                        if us < best[0]:
                            best = (us, names[v])
                    lib.lx_tune_clear()
                    gbs_best = nbytes / (best[0] * 1e-6) / 1e9
                    gbs_heur = nbytes / (heur * 1e-6) / 1e9
                    rows.append((fmt, N, K, M, gbs_heur, best[1], gbs_best))
                    print(
                        f"| {fmt} | {N} x {K} | {M} | heuristic {gbs_heur:.0f} GB/s ({100 * gbs_heur / B.DRAM_GBPS:.1f}%)"
                        f" | best {best[1]} {gbs_best:.0f} GB/s ({100 * gbs_best / B.DRAM_GBPS:.1f}%) |",
                        flush=True,
                    )
                del copies
                torch.cuda.empty_cache()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
