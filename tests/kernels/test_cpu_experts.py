"""The CPU expert kernel (engine/cpu/moe_experts.cpp, lx_moe_experts_decode on host memory) against
PyTorch, in every weight format, through both the AVX2 and the scalar paths.

The library is build/libdromos_cpu.so beside LX_KERNELS (or LX_CPU_KERNELS). Tensors are CPU tensors;
the reference dequantizes and computes in float64 with the intermediate rounded to bf16.
"""

import ctypes
import os

import pytest
import torch
import torch.nn.functional as F

from conftest import close

BF16, INT8, FP8, NVFP4 = 0, 1, 2, 3
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                    dtype=torch.float64)


class Weight(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("data", ctypes.c_void_p), ("scale", ctypes.c_void_p), ("glob", ctypes.c_float),
                ("expert_global", ctypes.c_void_p)]


@pytest.fixture(scope="module")
def cpu():
    path = os.environ.get("LX_CPU_KERNELS")
    if not path and os.environ.get("LX_KERNELS"):
        path = os.path.join(os.path.dirname(os.environ["LX_KERNELS"]), "libdromos_cpu.so")
    if not path or not os.path.exists(path):
        pytest.skip("libdromos_cpu.so not built")
    lib = ctypes.CDLL(path)
    lib.lx_moe_experts_decode.restype = ctypes.c_int
    lib.lx_moe_decode_workspace_bytes.restype = ctypes.c_int64
    lib.lx_moe_decode_workspace_bytes.argtypes = [ctypes.c_int] * 3
    lib.lx_cpu_force_scalar.argtypes = [ctypes.c_int]
    return lib


def quantize(w, fmt):
    """w [..., rows, cols] f32 -> (data, scale, global, dequantized float64)."""
    if fmt == BF16:
        q = w.to(torch.bfloat16)
        return q, None, 1.0, q.double()
    absmax = w.abs().amax(-1)
    if fmt == INT8:
        s = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
        q = torch.round(w / s.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        return q, s.float(), 1.0, q.double() * s.double().unsqueeze(-1)
    if fmt == FP8:
        s = torch.where(absmax > 0, absmax / 448.0, torch.ones_like(absmax))
        q = (w / s.unsqueeze(-1)).to(torch.float8_e4m3fn)
        return q, s.float(), 1.0, q.double() * s.double().unsqueeze(-1)
    # NVFP4: random codes, e4m3 block scales, one global
    g = torch.Generator().manual_seed(int(w.numel()) % 1000)
    codes = torch.randint(0, 16, w.shape, generator=g, dtype=torch.uint8)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
    bs = (torch.rand(*w.shape[:-1], w.shape[-1] // 16, generator=g) * 2 + 0.25).to(torch.float8_e4m3fn)
    glob = 0.01
    deq = E2M1[codes.long()] * bs.double().repeat_interleave(16, -1) * glob
    return packed, bs, glob, deq


def run(cpu, latent, ids, weights, gu, dn, L, I, scalar):
    T, K = ids.shape
    out = torch.full((T, L), float("nan"))
    ws = torch.zeros(cpu.lx_moe_decode_workspace_bytes(T, I, K), dtype=torch.uint8)
    cpu.lx_cpu_force_scalar(1 if scalar else 0)
    keep = (latent, ids, weights)
    status = cpu.lx_moe_experts_decode(
        ctypes.c_void_p(latent.data_ptr()), ctypes.c_void_p(ids.data_ptr()), ctypes.c_void_p(weights.data_ptr()),
        ctypes.byref(gu), ctypes.byref(dn), ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(ws.data_ptr()),
        T, L, I, K, None)
    cpu.lx_cpu_force_scalar(0)
    assert status == 0
    del keep
    return out


def reference(latent, ids, weights, gu_deq, dn_deq, I):
    T, K = ids.shape
    x = latent.double()
    out = torch.zeros(T, gu_deq.shape[-1], dtype=torch.float64)
    for t in range(T):
        for k in range(K):
            e = int(ids[t, k])
            mid = (F.silu(x[t] @ gu_deq[e, :I].T) * (x[t] @ gu_deq[e, I:].T)).to(torch.bfloat16).double()
            out[t] += float(weights[t, k]) * (mid @ dn_deq[e].T)
    return out


@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("fmt", [BF16, INT8, FP8, NVFP4])
@pytest.mark.parametrize("T,E,L,I,K", [(1, 8, 1024, 768, 4), (2, 6, 1024, 512, 4), (8, 12, 256, 64, 8),
                                       (1, 3, 96, 48, 1), (3, 5, 1040, 528, 2)])
def test_cpu_experts(cpu, fmt, T, E, L, I, K, scalar):
    if fmt == NVFP4 and (L % 16 or I % 16):
        pytest.skip("NVFP4 needs multiples of 16")
    g = torch.Generator().manual_seed(T * 31 + E + L + I + K + fmt)
    gu_w = torch.randn(E, 2 * I, L, generator=g) / L**0.5
    dn_w = torch.randn(E, L, I, generator=g) / I**0.5
    gq, gs, gg, gdeq = quantize(gu_w, fmt)
    dq, ds, dg, ddeq = quantize(dn_w, fmt)
    gq, dq = gq.contiguous(), dq.contiguous()
    gs = gs.contiguous() if gs is not None else None
    ds = ds.contiguous() if ds is not None else None
    gu = Weight(fmt, gq.data_ptr(), gs.data_ptr() if gs is not None else None, gg)
    dn = Weight(fmt, dq.data_ptr(), ds.data_ptr() if ds is not None else None, dg)
    ids = torch.rand(T, E, generator=g).argsort(-1)[:, :K].to(torch.int32).contiguous()
    weights = torch.softmax(torch.randn(T, K, generator=g), -1).contiguous()
    latent = torch.randn(T, L, generator=g).to(torch.bfloat16).contiguous()
    out = run(cpu, latent, ids, weights, gu, dn, L, I, scalar)
    want = reference(latent, ids, weights, gdeq, ddeq, I)
    close(out, want.float(), atol=1e-3 * want.abs().max().item(), rtol=1e-3)


def test_cpu_matches_between_paths(cpu):
    """The AVX2 and scalar paths agree to f32 rounding on a Logos-sized FP8 expert."""
    g = torch.Generator().manual_seed(5)
    E, L, I, K = 4, 1024, 768, 4
    gq, gs, _, _ = quantize(torch.randn(E, 2 * I, L, generator=g) / 32, FP8)
    dq, ds, _, _ = quantize(torch.randn(E, L, I, generator=g) / 28, FP8)
    gu = Weight(FP8, gq.data_ptr(), gs.data_ptr(), 1.0)
    dn = Weight(FP8, dq.data_ptr(), ds.data_ptr(), 1.0)
    ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    weights = torch.softmax(torch.randn(1, K, generator=g), -1)
    latent = torch.randn(1, L, generator=g).to(torch.bfloat16)
    a = run(cpu, latent, ids, weights, gu, dn, L, I, False)
    b = run(cpu, latent, ids, weights, gu, dn, L, I, True)
    close(a, b, atol=1e-4 * b.abs().max().item(), rtol=1e-4)
