#!/usr/bin/env python3
"""One lx_gemm_w call (after a warm-up), for profilers: python bench/one_gemm.py PAIR M N K [VARIANT]"""
import ctypes, os, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "kernels"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wformats import quantize_weight  # noqa: E402
from bench_gemm_w import PAIRS  # noqa: E402
from tune_gemm_w import FAMILY  # noqa: E402

pair, M, N, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
lib = ctypes.CDLL(os.environ["LX_KERNELS"])
lib.lx_gemm_workspace_bytes.restype = ctypes.c_int64
if len(sys.argv) > 5:
    lib.lx_tune_set(("gemm_w." + FAMILY[pair]).encode(), M, N, K, int(sys.argv[5]))
wt = quantize_weight(torch.randn(N, K, device="cuda") / K**0.5, PAIRS[pair][0])
math = PAIRS[pair][1]
x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
y = torch.empty(M, N, device="cuda")
ws = torch.empty(max(lib.lx_gemm_workspace_bytes(M, K, math), 1), dtype=torch.uint8, device="cuda")
args = (ctypes.c_void_p(x.data_ptr()), 1, K, ctypes.byref(wt.struct), math, None, 0, ctypes.c_void_p(y.data_ptr()), 0, N,
        M, N, K, ctypes.c_float(1.0), ctypes.c_float(0.0), ctypes.c_void_p(ws.data_ptr()),
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
for _ in range(3):
    assert lib.lx_gemm_w(*args) == 0
torch.cuda.synchronize()
