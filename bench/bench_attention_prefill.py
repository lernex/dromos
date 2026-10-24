#!/usr/bin/env python3
"""lx_attention_varlen on one prefill (or a continuation), Praxis and Logos heads: milliseconds per call
and TFLOPS counted as the causal 4 * n * L_avg * H * Dh. Long sequences take the tensor-core kernel.
Run on the GPU server: scripts/remote.sh SLOT x '$PY bench/bench_attention_prefill.py'"""
import ctypes
import math
import os

import torch

CASES = [  # (name, H, KVH, Dh, start, n)
    ("praxis", 32, 8, 64, 0, 512), ("praxis", 32, 8, 64, 0, 2048), ("praxis", 32, 8, 64, 0, 8192),
    ("praxis", 32, 8, 64, 6144, 2048), ("logos", 40, 10, 64, 0, 2048), ("logos", 40, 10, 64, 0, 8192),
]


def main():
    lib = ctypes.CDLL(os.environ["LX_KERNELS"])
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    p = lambda t: ctypes.c_void_p(t.data_ptr())
    print(f"{'model':8s} {'start':>6s} {'n':>6s} {'ms':>9s} {'TFLOPS':>8s}")
    for name, H, KVH, Dh, start, n in CASES:
        width = (H + 2 * KVH) * Dh
        qkv = torch.randn(n, width, device="cuda")
        cap = start + n
        k = torch.randn(1, cap, KVH, Dh, device="cuda").to(torch.bfloat16)
        v = torch.randn(1, cap, KVH, Dh, device="cuda").to(torch.bfloat16)
        out = torch.empty(n, H * Dh, device="cuda", dtype=torch.bfloat16)
        i32 = lambda x: torch.tensor(x, dtype=torch.int32, device="cuda")
        slots, starts, cu = i32([0]), i32([start]), i32([0, n])
        call = lambda: lib.lx_attention_varlen(p(qkv), width, p(k), p(v), p(slots), p(starts), p(cu), 1, cap, p(out), H,
                                               KVH, Dh, ctypes.c_float(1 / math.sqrt(Dh)), stream)
        assert call() == 0
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(3):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(10):
                call()
            e1.record()
            torch.cuda.synchronize()
            best = min(best, e0.elapsed_time(e1) / 10)
        flops = 4.0 * n * (start + (n + 1) / 2) * H * Dh
        print(f"{name:8s} {start:6d} {n:6d} {best:9.3f} {flops / best / 1e9:8.1f}", flush=True)


if __name__ == "__main__":
    main()
