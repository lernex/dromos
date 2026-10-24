"""Differential check of the int8 decode GEMV and the int8 MoE: the CUDA library against the HIP library
(NVIDIA shim). Weights are quantized once, by the CUDA library, so both libraries see identical bytes.

    $PY proto/diff_gemv.py build/liblernex_kernels.so build/e2e/hip/liblernex_hip.so
"""

import ctypes
import sys

import torch

cuda_lib = ctypes.CDLL(sys.argv[1])
hip_lib = ctypes.CDLL(sys.argv[2])
vp, ci, cf = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
LX_F32, LX_BF16 = 0, 1
LX_W_BF16, LX_W_INT8 = 0, 1


def bind(lib, name, argtypes):
    f = getattr(lib, name)
    f.argtypes = argtypes
    f.restype = ctypes.c_int
    return f


gemv_args = [vp, ci, ci, vp, ci, vp, vp, ci, vp, ci, ci, ci, ci, ci, cf, cf, vp]
moe_args = [vp, vp, vp, vp, vp, vp, vp, vp, vp, ci, ci, ci, ci, vp]


def ptr(t):
    return vp(t.data_ptr()) if t is not None else vp(None)


def report(label, a, b):
    d = (a.float() - b.float()).abs()
    print(f"{label:38s} max|diff| {d.max().item():.3g}  mean|diff| {d.mean().item():.3g}  "
          f"max|out| {a.float().abs().max().item():.3g}  differing {(d > 0).sum().item()} of {d.numel()}")


cu_gemv = bind(cuda_lib, "lx_gemv", gemv_args)
hp_gemv = bind(hip_lib, "lx_gemv", gemv_args)
cu_quant = bind(cuda_lib, "lx_quantize_w8", [vp, vp, vp, ctypes.c_int64, ci, vp])
hp_quant = bind(hip_lib, "lx_quantize_w8", [vp, vp, vp, ctypes.c_int64, ci, vp])
cu_moe = bind(cuda_lib, "lx_moe_experts_w8", moe_args)
hp_moe = bind(hip_lib, "lx_moe_experts_w8", moe_args)
cu_moe_ws = cuda_lib.lx_moe_workspace_bytes
cu_moe_ws.argtypes = [ci, ci, ci]
cu_moe_ws.restype = ctypes.c_int64
hp_moe_ws = hip_lib.lx_moe_workspace_bytes
hp_moe_ws.argtypes = [ci, ci, ci]
hp_moe_ws.restype = ctypes.c_int64


def quantize(w2d):
    """w2d: bf16 [rows, cols] on the device. Quantized by the CUDA library, checked against the HIP one."""
    rows, cols = w2d.shape
    q_a = torch.empty(rows, cols, device="cuda", dtype=torch.int8)
    s_a = torch.empty(rows, device="cuda")
    assert cu_quant(ptr(w2d), ptr(q_a), ptr(s_a), rows, cols, None) == 0
    q_b = torch.empty_like(q_a)
    s_b = torch.empty_like(s_a)
    assert hp_quant(ptr(w2d), ptr(q_b), ptr(s_b), rows, cols, None) == 0
    torch.cuda.synchronize()
    print(f"quantize {rows}x{cols}: int8 bitwise {torch.equal(q_a, q_b)}, scale bitwise {torch.equal(s_a, s_b)}")
    return q_a, s_a


print("== lx_gemv int8 weights (decode, M <= 8), bf16 x, f32 y")
for N, K in [(2048, 2048), (6144, 2048), (2048, 6144), (512, 2048), (64, 300)]:
    torch.manual_seed(0)
    w = (torch.randn(N, K, device="cuda") * 0.05).to(torch.bfloat16)
    q, s = quantize(w)
    for M in [1, 2, 4, 8]:
        x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        ya = torch.zeros(M, N, device="cuda")
        yb = torch.zeros(M, N, device="cuda")
        assert cu_gemv(ptr(x), LX_BF16, K, ptr(q), LX_W_INT8, ptr(s), None, LX_BF16, ptr(ya), LX_F32, N, M, N, K,
                       1.0, 0.0, None) == 0
        assert hp_gemv(ptr(x), LX_BF16, K, ptr(q), LX_W_INT8, ptr(s), None, LX_BF16, ptr(yb), LX_F32, N, M, N, K,
                       1.0, 0.0, None) == 0
        torch.cuda.synchronize()
        report(f"gemv int8 N{N} K{K} M{M}", ya, yb)

print("== lx_moe_experts_w8 (int8 experts, bf16 latent, f32 out)")
E, top_k = 8, 2
for T, L, I in [(1, 2048, 512), (4, 2048, 512), (8, 2048, 512), (8, 1024, 300), (300, 2048, 512)]:
    torch.manual_seed(0)
    gu = (torch.randn(E * 2 * I, L, device="cuda") * 0.05).to(torch.bfloat16)
    dn = (torch.randn(E * L, I, device="cuda") * 0.05).to(torch.bfloat16)
    gu_q, gu_s = quantize(gu)
    dn_q, dn_s = quantize(dn)
    gu_q, gu_s = gu_q.view(E, 2 * I, L), gu_s.view(E, 2 * I)
    dn_q, dn_s = dn_q.view(E, L, I), dn_s.view(E, L)
    latent = torch.randn(T, L, device="cuda").to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device="cuda")[:top_k] for _ in range(T)]).to(torch.int32)
    weights = torch.softmax(torch.randn(T, top_k, device="cuda"), dim=-1)
    ws_bytes = max(cu_moe_ws(T, I, top_k), hp_moe_ws(T, I, top_k))
    ws = torch.empty(max(ws_bytes, 1), device="cuda", dtype=torch.uint8)
    oa = torch.empty(T, L, device="cuda")
    ob = torch.empty(T, L, device="cuda")
    assert cu_moe(ptr(latent), ptr(ids), ptr(weights), ptr(gu_q), ptr(gu_s), ptr(dn_q), ptr(dn_s), ptr(oa), ptr(ws),
                  T, L, I, top_k, None) == 0
    assert hp_moe(ptr(latent), ptr(ids), ptr(weights), ptr(gu_q), ptr(gu_s), ptr(dn_q), ptr(dn_s), ptr(ob), ptr(ws),
                  T, L, I, top_k, None) == 0
    torch.cuda.synchronize()
    report(f"moe_w8 T{T} L{L} I{I}", oa, ob)
