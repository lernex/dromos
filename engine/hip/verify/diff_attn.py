"""Differential check of lx_attention_varlen: the CUDA library (cluster split mode for few items) against
the HIP library (one block per item). Same qkv, same caches, both outputs and caches compared.

    $PY proto/diff_attn.py build/libdromos_kernels.so build/e2e/hip/libdromos_hip.so
"""

import ctypes
import math
import sys

import torch

cuda_lib = ctypes.CDLL(sys.argv[1])
hip_lib = ctypes.CDLL(sys.argv[2])
vp, ci, cf = ctypes.c_void_p, ctypes.c_int, ctypes.c_float


def attn_fn(lib):
    f = lib.lx_attention_varlen
    f.argtypes = [vp, ci, vp, vp, vp, vp, vp, ci, ci, vp, ci, ci, ci, cf, vp]
    f.restype = ctypes.c_int
    return f


def ptr(t):
    return vp(t.data_ptr())


def run(f, qkv, kc0, vc0, slots, starts, cu, n_seq, capacity, H, KVH, Dh, T):
    kc, vc = kc0.clone(), vc0.clone()
    out = torch.empty(T, H * Dh, device="cuda", dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(Dh)
    rc = f(ptr(qkv), qkv.shape[1], ptr(kc), ptr(vc), ptr(slots), ptr(starts), ptr(cu), n_seq, capacity,
           ptr(out), H, KVH, Dh, scale, None)
    assert rc == 0, rc
    torch.cuda.synchronize()
    return out, kc, vc


def report(label, a, b):
    d = (a.float() - b.float()).abs()
    print(f"{label:30s} max|diff| {d.max().item():.3g}  mean|diff| {d.mean().item():.3g}  "
          f"max|out| {a.float().abs().max().item():.3g}  differing {(d > 0).sum().item()} of {d.numel()}")


cu_f, hp_f = attn_fn(cuda_lib), attn_fn(hip_lib)
H, KVH, Dh, capacity = 16, 8, 128, 4096
ld = (H + 2 * KVH) * Dh

cases = [
    # (label, per-sequence new token counts, start positions)
    ("decode 1 seq @500", [1], [500]),
    ("decode 4 seqs @100", [1, 1, 1, 1], [100, 100, 100, 100]),
    ("decode 8 seqs @3000", [1] * 8, [3000] * 8),
    ("decode 1 seq @0", [1], [0]),
    ("decode 2 seqs @4000", [1, 1], [4000, 4000]),
    ("prefill 2x200 (no split)", [200, 200], [0, 0]),
]

for label, counts, starts_list in cases:
    torch.manual_seed(0)
    n_seq = len(counts)
    T = sum(counts)
    slots = torch.arange(n_seq, dtype=torch.int32, device="cuda")
    starts = torch.tensor(starts_list, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0] + list(torch.tensor(counts).cumsum(0).tolist()), dtype=torch.int32, device="cuda")
    qkv = torch.randn(T, ld, device="cuda")
    kc0 = (torch.randn(n_seq, capacity, KVH, Dh, device="cuda") * 0.5).to(torch.bfloat16)
    vc0 = (torch.randn(n_seq, capacity, KVH, Dh, device="cuda") * 0.5).to(torch.bfloat16)
    a_out, a_k, a_v = run(cu_f, qkv, kc0, vc0, slots, starts, cu, n_seq, capacity, H, KVH, Dh, T)
    b_out, b_k, b_v = run(hp_f, qkv, kc0, vc0, slots, starts, cu, n_seq, capacity, H, KVH, Dh, T)
    report(f"{label}: out", a_out, b_out)
    report(f"{label}: k_cache", a_k, b_k)
    report(f"{label}: v_cache", a_v, b_v)
