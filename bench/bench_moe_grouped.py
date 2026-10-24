#!/usr/bin/env python3
"""lx_moe_experts_grouped per weight format and math at Praxis's experts (E = 128, top-4, L = 1024,
I = 512; MODEL=logos for E = 192, I = 768), T = 16 to 8192 tokens, uniformly routed. Each figure is the
whole call: routing sort, quantization, both GEMMs and the sum. Columns: microseconds per call and
TFLOPS counted as 6 * T * top_k * L * I (the two GEMMs); the bf16 lx_moe_experts kernel for reference.

Timing: CUDA events around back-to-back calls (enough for ~20 ms), best of 3.
Environment: LX_KERNELS; PAIRS (default all); TS (comma list); MODEL (praxis | logos);
LX_MOE_ORDERED=1 to time the deterministic sum.
Run on the GPU server: scripts/remote.sh SLOT x '$PY bench/bench_moe_grouped.py'
"""

import ctypes
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "kernels"))
from wformats import quantize_weight  # noqa: E402
from bench_gemm_w import PAIRS  # noqa: E402

MODELS = {"praxis": (1024, 512, 128, 4), "logos": (1024, 768, 192, 4)}
TS = [16, 64, 256, 1024, 2048, 4096, 8192]


def timed(fn, flops):
    fn()
    torch.cuda.synchronize()
    iters = max(3, min(500, int(0.02 / max(flops / 100e12, 2e-6))))
    best = float("inf")
    for _ in range(3):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / 1e3 / iters)
    return best


def main():
    lib = ctypes.CDLL(os.environ["LX_KERNELS"])
    lib.lx_moe_grouped_workspace_bytes.restype = ctypes.c_int64
    L, I, E, top_k = MODELS[os.environ.get("MODEL", "praxis")]
    pairs = os.environ.get("PAIRS", ",".join(PAIRS)).split(",")
    ts = [int(t) for t in os.environ.get("TS", ",".join(map(str, TS))).split(",")]
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    p = lambda t: ctypes.c_void_p(t.data_ptr())
    gu_src = torch.randn(E, 2 * I, L, device="cuda") / L**0.5
    dn_src = torch.randn(E, L, I, device="cuda") / I**0.5
    gu_bf = gu_src.to(torch.bfloat16)
    dn_bf = dn_src.to(torch.bfloat16)
    print(f"L={L} I={I} E={E} top_k={top_k} ordered={os.environ.get('LX_MOE_ORDERED', '0')}")
    print(f"{'T':>5s} " + " ".join(f"{q:>14s}" for q in ["lx_moe_experts"] + pairs) + "   (us | TFLOPS)")
    for T in ts:
        latent = torch.randn(T, L, device="cuda").to(torch.bfloat16)
        ids = torch.rand(T, E, device="cuda").argsort(-1)[:, :top_k].to(torch.int32).contiguous()
        weights = torch.softmax(torch.randn(T, top_k, device="cuda"), -1).contiguous()
        out = torch.empty(T, L, device="cuda")
        flops = 6.0 * T * top_k * L * I
        row = []
        old = lambda: lib.lx_moe_experts(p(latent), p(ids), p(weights), p(gu_bf), p(dn_bf), p(out), T, L, I, top_k, stream)
        sec = timed(old, flops)
        row.append(f"{sec * 1e6:8.0f}|{flops / sec / 1e12:5.1f}")
        for name in pairs:
            wtype, math = PAIRS[name]
            gu, dn = quantize_weight(gu_src, wtype), quantize_weight(dn_src, wtype)
            ws = torch.empty(lib.lx_moe_grouped_workspace_bytes(T, L, I, E, top_k, math), dtype=torch.uint8, device="cuda")
            call = lambda: lib.lx_moe_experts_grouped(p(latent), p(ids), p(weights), ctypes.byref(gu.struct),
                                                      ctypes.byref(dn.struct), math, p(out), p(ws), T, L, I, E, top_k,
                                                      stream)
            status = call()
            if status != 0:
                row.append(f"{'err' + str(status):>14s}")
                continue
            sec = timed(call, flops)
            row.append(f"{sec * 1e6:8.0f}|{flops / sec / 1e12:5.1f}")
            del gu, dn, ws
        print(f"{T:5d} " + " ".join(f"{c:>14s}" for c in row), flush=True)


if __name__ == "__main__":
    main()
