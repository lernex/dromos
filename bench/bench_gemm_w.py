#!/usr/bin/env python3
"""TFLOPS of lx_gemm_w per weight format and math at Praxis's prefill shapes, M = 64 to 8192, next to
cuBLASLt BF16 (lx_gemm_bf16, the bf16 column). Each figure is the whole lx_gemm_w call, activation
quantization included, as the model pays it: bf16 x in, f32 y out.

Timing: CUDA events around a run of back-to-back calls (enough for ~20 ms), best of 3 runs.
Environment: LX_KERNELS (set by scripts/remote.sh), SHAPES (comma list of names, default all),
PAIRS (comma list, default all), MS (comma list of M).

Run on the GPU server: scripts/remote.sh SLOT x '$PY bench/bench_gemm_w.py'
"""

import ctypes
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "kernels"))
from wformats import MATH_BF16, MATH_FP4, MATH_FP8, MATH_INT8, W_BF16, W_FP8, W_INT8, W_NVFP4, quantize_weight  # noqa: E402

SHAPES = {  # name: (N, K, rows per token)
    "in_proj": (10304, 2048, 1),
    "out_proj": (2048, 4096, 1),
    "qkv": (3072, 2048, 1),
    "attn_out": (2048, 2048, 1),
    "latent_down": (1024, 2048, 1),
    "latent_up": (2048, 1024, 1),
    "m_query": (256, 2048, 4),
    "m_route": (256, 6400, 1),
}
PAIRS = {  # name: (weight format, math)
    "bf16": (W_BF16, MATH_BF16),
    "w8a16": (W_INT8, MATH_BF16),
    "w8a16f": (W_FP8, MATH_BF16),
    "w4a16": (W_NVFP4, MATH_BF16),
    "int8": (W_INT8, MATH_INT8),
    "fp8": (W_FP8, MATH_FP8),
    "w4a8": (W_NVFP4, MATH_FP8),
    "fp4": (W_NVFP4, MATH_FP4),
}
MS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


def main():
    lib = ctypes.CDLL(os.environ["LX_KERNELS"])
    lib.lx_gemm_workspace_bytes.restype = ctypes.c_int64
    shapes = os.environ.get("SHAPES", ",".join(SHAPES)).split(",")
    pairs = os.environ.get("PAIRS", ",".join(PAIRS)).split(",")
    ms = [int(m) for m in os.environ.get("MS", ",".join(map(str, MS))).split(",")]
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    print(f"{'shape':12s} {'M':>5s} " + " ".join(f"{p:>7s}" for p in pairs) + "   (TFLOPS)")
    for name in shapes:
        N, K, rpt = SHAPES[name]
        w_src = torch.randn(N, K, device="cuda") / K**0.5
        weights = {p: quantize_weight(w_src, PAIRS[p][0]) for p in pairs}
        for M0 in ms:
            M = M0 * rpt
            x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
            y = torch.empty(M, N, device="cuda")
            row = []
            for p in pairs:
                wt, math = weights[p], PAIRS[p][1]
                ws = torch.empty(max(lib.lx_gemm_workspace_bytes(M, K, math), 1), dtype=torch.uint8, device="cuda")
                args = (ctypes.c_void_p(x.data_ptr()), 1, K, ctypes.byref(wt.struct), math, None, 0,
                        ctypes.c_void_p(y.data_ptr()), 0, N, M, N, K, ctypes.c_float(1.0), ctypes.c_float(0.0),
                        ctypes.c_void_p(ws.data_ptr()), stream)
                status = lib.lx_gemm_w(*args)
                if status != 0:
                    row.append(f"{'err' + str(status):>7s}")
                    continue
                torch.cuda.synchronize()
                flops = 2.0 * M * N * K
                iters = max(3, min(2000, int(0.02 / (flops / 100e12))))
                best = float("inf")
                for _ in range(3):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    for _ in range(iters):
                        lib.lx_gemm_w(*args)
                    e1.record()
                    torch.cuda.synchronize()
                    best = min(best, e0.elapsed_time(e1) / 1e3 / iters)
                row.append(f"{flops / best / 1e12:7.1f}")
            print(f"{name:12s} {M:5d} " + " ".join(row), flush=True)


if __name__ == "__main__":
    main()
