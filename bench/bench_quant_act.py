#!/usr/bin/env python3
"""lx_quantize_act throughput (GB/s of activations read) at the shapes prefill quantizes.
Run on the GPU server: scripts/remote.sh SLOT x '$PY bench/bench_quant_act.py'"""
import ctypes
import os

import torch

MATHS = {"int8": 1, "fp8": 2, "fp4": 3}
SHAPES = [(2048, 2048, torch.bfloat16), (8192, 2048, torch.bfloat16), (8192, 4096, torch.bfloat16),
          (32768, 512, torch.bfloat16), (8192, 1024, torch.float32), (8192, 6400, torch.float32)]


def main():
    lib = ctypes.CDLL(os.environ["LX_KERNELS"])
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    p = lambda t: ctypes.c_void_p(t.data_ptr())
    print(f"{'M':>6s} {'K':>5s} {'x':>5s} " + " ".join(f"{m:>14s}" for m in MATHS) + "   (us | GB/s read)")
    for M, K, dt in SHAPES:
        x = torch.randn(M, K, device="cuda").to(dt)
        q = torch.empty(M, K, dtype=torch.uint8, device="cuda")
        s = torch.empty(M, device="cuda")
        bs = torch.empty(M, K // 16, dtype=torch.uint8, device="cuda")
        row = []
        for name, math in MATHS.items():
            call = lambda: lib.lx_quantize_act(p(x), 0 if dt == torch.float32 else 1, K, math, p(q), K, p(s), p(bs), K // 16,
                                               M, K, stream)
            assert call() == 0
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(3):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(50):
                    call()
                e1.record()
                torch.cuda.synchronize()
                best = min(best, e0.elapsed_time(e1) / 1e3 / 50)
            row.append(f"{best * 1e6:7.1f}|{x.numel() * x.element_size() / best / 1e9:6.0f}")
        print(f"{M:6d} {K:5d} {str(dt)[6:]:>5s} " + " ".join(f"{c:>14s}" for c in row), flush=True)


if __name__ == "__main__":
    main()
