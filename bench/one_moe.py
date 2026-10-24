#!/usr/bin/env python3
"""A few lx_moe_experts_grouped calls (after a warm-up), for profilers: python bench/one_moe.py PAIR T [MODEL]"""
import ctypes, os, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "kernels"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wformats import quantize_weight  # noqa: E402
from bench_gemm_w import PAIRS  # noqa: E402
from bench_moe_grouped import MODELS  # noqa: E402

pair, T = sys.argv[1], int(sys.argv[2])
L, I, E, top_k = MODELS[sys.argv[3] if len(sys.argv) > 3 else "praxis"]
lib = ctypes.CDLL(os.environ["LX_KERNELS"])
lib.lx_moe_grouped_workspace_bytes.restype = ctypes.c_int64
wtype, math = PAIRS[pair]
gu = quantize_weight(torch.randn(E, 2 * I, L, device="cuda") / L**0.5, wtype)
dn = quantize_weight(torch.randn(E, L, I, device="cuda") / I**0.5, wtype)
latent = torch.randn(T, L, device="cuda").to(torch.bfloat16)
ids = torch.rand(T, E, device="cuda").argsort(-1)[:, :top_k].to(torch.int32).contiguous()
weights = torch.softmax(torch.randn(T, top_k, device="cuda"), -1).contiguous()
out = torch.empty(T, L, device="cuda")
ws = torch.empty(lib.lx_moe_grouped_workspace_bytes(T, L, I, E, top_k, math), dtype=torch.uint8, device="cuda")
p = lambda t: ctypes.c_void_p(t.data_ptr())
for _ in range(5):
    assert lib.lx_moe_experts_grouped(p(latent), p(ids), p(weights), ctypes.byref(gu.struct), ctypes.byref(dn.struct), math,
                                      p(out), p(ws), T, L, I, E, top_k,
                                      ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)) == 0
torch.cuda.synchronize()
