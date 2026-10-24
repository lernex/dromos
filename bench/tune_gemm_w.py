#!/usr/bin/env python3
"""Every registered tile variant of the lx_gemm_w families at the given shapes, in TFLOPS (the whole
call, quantization included), so the heuristic and the tuner's defaults can be checked against the
measured best. The engine's autotuner (runtime/tuner) does the same timing at load.

Environment: LX_KERNELS; PAIRS (default int8,fp8,fp4,w4a8,w8a16,w4a16); SHAPES as N:K pairs
(default the Praxis in_proj, out_proj, latent_down); MS (default 256,1024,8192).
Run on the GPU server: scripts/remote.sh SLOT x '$PY bench/tune_gemm_w.py'
"""

import ctypes
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "kernels"))
from wformats import quantize_weight  # noqa: E402
from bench_gemm_w import PAIRS  # noqa: E402

FAMILY = {"int8": "i8.sm80", "fp8": "f8.sm120", "fp4": "f4.sm120", "w4a8": "w4a8.sm120", "w8a16": "w8i8.sm80",
          "w8a16f": "w8f8.sm120", "w4a16": "w4a16.sm80", "bf16": "bf16.sm80"}


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    lib = ctypes.CDLL(os.environ["LX_KERNELS"])
    lib.lx_gemm_workspace_bytes.restype = ctypes.c_int64
    lib.lx_tune_variant_name.restype = ctypes.c_char_p
    pairs = os.environ.get("PAIRS", "int8,fp8,fp4,w4a8,w8a16,w4a16").split(",")
    shapes = [tuple(int(v) for v in s.split(":")) for s in os.environ.get("SHAPES", "10304:2048,2048:4096,1024:2048").split(",")]
    ms = [int(m) for m in os.environ.get("MS", "256,1024,8192").split(",")]
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    for p in pairs:
        fam = ("gemm_w." + FAMILY[p]).encode()
        nv = lib.lx_tune_variants(fam)
        names = [lib.lx_tune_variant_name(fam, v).decode() for v in range(nv)]
        print(f"== {p} ({fam.decode()})   " + " ".join(f"{n:>22s}" for n in names))
        for N, K in shapes:
            wt = quantize_weight(torch.randn(N, K, device="cuda") / K**0.5, PAIRS[p][0])
            math = PAIRS[p][1]
            for M in ms:
                x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
                y = torch.empty(M, N, device="cuda")
                ws = torch.empty(max(lib.lx_gemm_workspace_bytes(M, K, math), 1), dtype=torch.uint8, device="cuda")
                args = (ctypes.c_void_p(x.data_ptr()), 1, K, ctypes.byref(wt.struct), math, None, 0,
                        ctypes.c_void_p(y.data_ptr()), 0, N, M, N, K, ctypes.c_float(1.0), ctypes.c_float(0.0),
                        ctypes.c_void_p(ws.data_ptr()), stream)
                row = []
                flops = 2.0 * M * N * K
                for v in range(nv):
                    lib.lx_tune_set(fam, M, N, K, v)
                    if lib.lx_gemm_w(*args) != 0:
                        row.append(f"{'err':>22s}")
                        continue
                    torch.cuda.synchronize()
                    iters = max(3, min(2000, int(0.02 / (flops / 150e12))))
                    best = float("inf")
                    for _ in range(3):
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        for _ in range(iters):
                            lib.lx_gemm_w(*args)
                        e1.record()
                        torch.cuda.synchronize()
                        best = min(best, e0.elapsed_time(e1) / 1e3 / iters)
                    row.append(f"{flops / best / 1e12:22.1f}")
                lib.lx_tune_clear()
                print(f"  N={N:5d} K={K:5d} M={M:5d} " + " ".join(row), flush=True)


if __name__ == "__main__":
    main()
