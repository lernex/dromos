"""Differential check: the decode-path kernels of the CUDA library against the HIP library (NVIDIA shim).

Same inputs go to both libraries; the outputs are compared element by element. Run from the slot root:
    $PY proto/diff_decode.py build/libdromos_kernels.so build/e2e/hip/libdromos_hip.so
"""

import ctypes
import math
import sys

import torch

cuda_lib = ctypes.CDLL(sys.argv[1])
hip_lib = ctypes.CDLL(sys.argv[2])

vp, ci, cf = ctypes.c_void_p, ctypes.c_int, ctypes.c_float


def bind(lib, name, argtypes):
    f = getattr(lib, name)
    f.argtypes = argtypes
    f.restype = ctypes.c_int
    return f


def ptr(t):
    return vp(t.data_ptr())


def fuse_fn(lib):
    return bind(lib, "lx_memory_fuse", [vp, vp, vp, vp, vp, ci, ci, ci, vp])


def mhc_fn(lib):
    return bind(lib, "lx_mhc_read", [vp, vp, vp, vp, cf, vp, vp, ci, ci, ci, vp])


def attend_fn(lib):
    return bind(lib, "lx_memory_attend", [vp, vp, vp, ci, ci, vp, ci, ci, ci, vp])


def run_fuse(f, streams, projected, gate, gate_bias, T, S, D):
    s = streams.clone()
    summary = torch.empty(T, D, device="cuda")
    assert f(ptr(s), ptr(projected), ptr(gate), ptr(gate_bias), ptr(summary), T, S, D, None) == 0
    torch.cuda.synchronize()
    return s, summary


def run_mhc(f, streams, matrix, read, norm_w, eps, T, S, D):
    mixed = torch.empty(T, S, D, device="cuda")
    h = torch.empty(T, D, device="cuda", dtype=torch.bfloat16)
    assert f(ptr(streams), ptr(matrix), ptr(read), ptr(norm_w), eps, ptr(mixed), ptr(h), T, S, D, None) == 0
    torch.cuda.synchronize()
    return mixed, h


def run_attend(f, q, bank_k, bank_v, n, E, T, S, M):
    out = torch.empty(T, S, M, device="cuda", dtype=torch.bfloat16)
    assert f(ptr(q), ptr(bank_k), ptr(bank_v), n, E, ptr(out), T, S, M, None) == 0
    torch.cuda.synchronize()
    return out


def report(label, a, b):
    d = (a.float() - b.float()).abs()
    scale = a.float().abs().max().item()
    print(f"{label:34s} max|diff| {d.max().item():.3g}  mean|diff| {d.mean().item():.3g}  "
          f"max|out| {scale:.3g}  elements differing {(d > 0).sum().item()} of {d.numel()}")


fuse_cu, fuse_hp = fuse_fn(cuda_lib), fuse_fn(hip_lib)
mhc_cu, mhc_hp = mhc_fn(cuda_lib), mhc_fn(hip_lib)
att_cu, att_hp = attend_fn(cuda_lib), attend_fn(hip_lib)

print("== memory_fuse (streams f32 in place, summary f32)")
for T, S, D in [(8, 4, 2048), (1, 8, 2048), (5, 8, 300), (8, 3, 37), (300, 4, 2048), (1, 4, 9000)]:
    torch.manual_seed(0)
    streams = torch.randn(T, S, D, device="cuda")
    projected = torch.randn(T, S, D, device="cuda") * 0.5
    gate = torch.randn(S, D, device="cuda").to(torch.bfloat16)
    gate_bias = torch.randn(S, device="cuda").to(torch.bfloat16)
    a_s, a_m = run_fuse(fuse_cu, streams, projected, gate, gate_bias, T, S, D)
    b_s, b_m = run_fuse(fuse_hp, streams, projected, gate, gate_bias, T, S, D)
    report(f"fuse T{T} S{S} D{D} streams", a_s, b_s)
    report(f"fuse T{T} S{S} D{D} summary", a_m, b_m)

print("== mhc_read (mixed f32, h bf16)")
for T, S, D in [(8, 4, 2560), (1, 8, 2048), (4, 4, 2048), (300, 4, 2048), (1, 4, 9000)]:
    torch.manual_seed(0)
    streams = torch.randn(T, S, D, device="cuda")
    matrix = torch.randn(S, S, device="cuda")
    read = torch.randn(S, device="cuda")
    norm_w = (torch.rand(D, device="cuda") + 0.5).to(torch.bfloat16)
    a_mix, a_h = run_mhc(mhc_cu, streams, matrix, read, norm_w, 1e-6, T, S, D)
    b_mix, b_h = run_mhc(mhc_hp, streams, matrix, read, norm_w, 1e-6, T, S, D)
    report(f"mhc T{T} S{S} D{D} mixed", a_mix, b_mix)
    report(f"mhc T{T} S{S} D{D} h", a_h, b_h)

print("== memory_attend (out bf16)")
for T, S, M, E, n in [(8, 4, 256, 8, 8), (7, 2, 64, 8, 8), (1, 8, 256, 8, 8), (300, 4, 256, 8, 8)]:
    torch.manual_seed(0)
    q = torch.randn(T, S, M, device="cuda")
    bank_k = torch.randn(T, E, M, device="cuda")
    bank_v = torch.randn(T, E, M, device="cuda")
    bank_k[:, n:] = 0.0
    bank_v[:, n:] = 0.0
    a = run_attend(att_cu, q, bank_k, bank_v, n, E, T, S, M)
    b = run_attend(att_hp, q, bank_k, bank_v, n, E, T, S, M)
    report(f"attend T{T} S{S} M{M} n{n}", a, b)
