#!/usr/bin/env python3
"""Few-token expert kernels (lx_moe_experts_decode) on every weight format, at Metis shapes.

E = 128 experts, L = 1024, I = 512, top_k = 4, T in {1, 2, 4, 8}. Each format is a stack of random
weights of that format: BF16, INT8 and FP8 with per-row f32 scales, NVFP4 with e4m3 block scales and a
per-expert global, and the lossless BF16X encoding of the BF16 stacks. The lossless row goes through
lx_moe_experts_decode (which routes it to lx_moe_experts_bf16x); the lead's plain-bf16 kernel is the
baseline the lossless format is measured against.

Every timed call reads its expert tables from DRAM. A ~256 MB float read runs before each call and
evicts the 64 MB L2, and each call gets its own random routing, drawn ahead of time. CUDA events
bracket each call alone. Reported: the trimmed mean of those per-call times (the fastest and slowest
tenth are dropped), the bytes the distinct experts of a call move, and that as a share of the 896 GB/s
the card is rated for.

Environment: LX_KERNELS (the built library, set by scripts/remote.sh), N (calls per row, default 200).
Run on the GPU server: scripts/remote.sh formats x '$PY bench/bench_moe.py'
"""

import ctypes
import os
import random
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "kernels"))
from formats import LxWeight, W_BF16, W_FP8, W_INT8, W_NVFP4  # noqa: E402

W_BF16X = 4
E, L, I, TOP_K = 128, 1024, 512, 4
PEAK_GBS = 896.0
TOKENS = [1, 2, 4, 8]
FLUSH_BYTES = 256 << 20


def call(fn, *args):
    conv = []
    for a in args:
        if isinstance(a, torch.Tensor):
            conv.append(ctypes.c_void_p(a.data_ptr()))
        elif isinstance(a, (ctypes._Pointer, ctypes.c_void_p)):
            conv.append(a)
        else:
            conv.append(ctypes.c_int(a))
    conv.append(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    status = fn(*conv)
    if status != 0:
        raise RuntimeError(f"kernel returned CUDA error {status}")


def routing(T, n):
    """n random routings of T tokens: top_k distinct experts per token, drawn on the CPU."""
    tables = [torch.rand(T, E).argsort(-1)[:, :TOP_K].to(torch.int32) for _ in range(n)]
    uniques = [torch.unique(t).numel() for t in tables]
    return [t.cuda().contiguous() for t in tables], uniques


def timed(run, n, flush):
    """Trimmed-mean microseconds per call over n calls, each after a cold-L2 flush.

    Events bracket each call alone. The event clock ticks in ~2 us steps, so a single reading is
    coarse; averaged over many calls with random phase the mean is unbiased, and trimming drops the
    outliers that other work on the shared card causes."""
    for i in range(3):  # warm-up: lazy module loading and allocator warm-up stay out of the sample
        run(i)
    torch.cuda.synchronize()
    starts, ends = [], []
    for i in range(n):
        # A float read of ~256 MB evicts L2 with clean lines. The length varies slightly per call: the
        # event clock ticks in ~2 us steps and an unvaried flush locks every call to one phase.
        flush[: flush.numel() - random.randrange(4096)].sum()
        starts.append(torch.cuda.Event(enable_timing=True))
        ends.append(torch.cuda.Event(enable_timing=True))
        starts[-1].record()
        run(i)
        ends[-1].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    cut = n // 10  # trimmed mean: drop the slowest and fastest tenth
    return statistics.mean(times[cut : n - cut])


def bf16x_stack(lib, w):
    """The lossless encoding of a bf16 stack [rows, K] (lx_bf16x_encode), as a device uint8 tensor."""
    rows, K = w.shape
    lib.lx_bf16x_bytes.restype = ctypes.c_int64
    lib.lx_bf16x_bytes.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p]
    n = lib.lx_bf16x_bytes(ctypes.c_void_p(w.data_ptr()), rows, K, None)
    assert n > 0, f"lx_bf16x_bytes returned {n}"
    out = torch.full((n,), 0xAA, dtype=torch.uint8, device="cuda")
    lib.lx_bf16x_encode.restype = ctypes.c_int
    lib.lx_bf16x_encode.argtypes = [
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
    ]
    status = lib.lx_bf16x_encode(ctypes.c_void_p(w.data_ptr()), rows, K, ctypes.c_void_p(out.data_ptr()), n, None)
    assert status == 0, f"lx_bf16x_encode returned {status}"
    return out


def make_stack(lib, fmt, rows, K):
    """A stack of `rows` rows of K values in format fmt (rows = E * N). Returns (LxWeight, tensors to keep)."""
    g = torch.Generator(device="cuda").manual_seed(rows + K)
    if fmt == "bf16":
        w = (torch.randn(rows, K, device="cuda", generator=g) / K**0.5).to(torch.bfloat16)
        return LxWeight(type=W_BF16, data=w.data_ptr(), scale=None, global_=1.0, expert_global=None), [w]
    if fmt == "int8":
        q = torch.randint(-127, 128, (rows, K), device="cuda", generator=g, dtype=torch.int32).to(torch.int8)
        s = (torch.rand(rows, device="cuda", generator=g) * 0.02 + 0.001).float()
        return LxWeight(type=W_INT8, data=q.data_ptr(), scale=s.data_ptr(), global_=1.0, expert_global=None), [q, s]
    if fmt == "fp8":
        b = (torch.randn(rows, K, device="cuda", generator=g) * 0.5).clamp(-448, 448).to(torch.float8_e4m3fn)
        b = b.view(torch.uint8)
        s = (torch.rand(rows, device="cuda", generator=g) * 0.02 + 0.001).float()
        return LxWeight(type=W_FP8, data=b.data_ptr(), scale=s.data_ptr(), global_=1.0, expert_global=None), [b, s]
    if fmt == "nvfp4":
        codes = torch.randint(0, 256, (rows, K // 2), device="cuda", generator=g, dtype=torch.int32).to(torch.uint8)
        blocks = (torch.rand(rows, K // 16, device="cuda", generator=g) * 2.0).to(torch.float8_e4m3fn).view(torch.uint8)
        per_expert = (0.01 + 0.004 * torch.arange(E, device="cuda")).float()
        return (
            LxWeight(type=W_NVFP4, data=codes.data_ptr(), scale=blocks.data_ptr(), global_=1.0,
                     expert_global=per_expert.data_ptr()),
            [codes, blocks, per_expert],
        )
    if fmt == "bf16x":
        w = (torch.randn(rows, K, device="cuda", generator=g) / K**0.5).to(torch.bfloat16)
        enc = bf16x_stack(lib, w)
        return LxWeight(type=W_BF16X, data=enc.data_ptr(), scale=None, global_=1.0, expert_global=None), [w, enc]
    raise ValueError(fmt)


def bytes_per_expert(lib, fmt):
    """Bytes one expert's gate/up and down tables move, in the format."""
    if fmt == "bf16":
        return 2 * (2 * I * L + L * I)
    if fmt in ("int8", "fp8"):
        return (2 * I * L + L * I) + 4 * (2 * I + L)
    if fmt == "nvfp4":
        return (2 * I * L + L * I) // 2 + (2 * I * L + L * I) // 16
    if fmt == "bf16x":
        lib.lx_bf16x_row_bytes.restype = ctypes.c_int64
        lib.lx_bf16x_row_bytes.argtypes = [ctypes.c_int]
        return 2 * I * lib.lx_bf16x_row_bytes(L) + L * lib.lx_bf16x_row_bytes(I)
    if fmt == "bf16 (lead)":
        return 2 * (2 * I * L + L * I)
    raise ValueError(fmt)


def main():
    lib = ctypes.CDLL(os.environ["LX_KERNELS"])
    lib.lx_moe_experts_decode.restype = ctypes.c_int
    lib.lx_moe_decode_workspace_bytes.restype = ctypes.c_int64
    lib.lx_moe_decode_workspace_bytes.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.lx_moe_experts_decode_bf16.restype = ctypes.c_int
    n_iter = int(os.environ.get("N", "200"))

    torch.manual_seed(0)
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    rows = []
    for fmt in ["bf16", "int8", "fp8", "nvfp4", "bf16x", "bf16 (lead)"]:
        if fmt == "bf16 (lead)":
            # The lead's plain-bf16 decode kernel, the baseline the lossless format is measured against.
            gu_bf, keep_gu = make_stack(lib, "bf16", E * 2 * I, L)
            dn_bf, keep_dn = make_stack(lib, "bf16", E * L, I)
            gu_ptr, dn_ptr = gu_bf.data, dn_bf.data
            kernel = lib.lx_moe_experts_decode_bf16
            keep = keep_gu + keep_dn
        else:
            gu_w, keep_gu = make_stack(lib, fmt, E * 2 * I, L)
            dn_w, keep_dn = make_stack(lib, fmt, E * L, I)
            keep = keep_gu + keep_dn
            kernel = lib.lx_moe_experts_decode
        per_expert = bytes_per_expert(lib, fmt)
        for T in TOKENS:
            n = n_iter
            ids_list, uniques = routing(T, n)
            latent = torch.randn(T, L, device="cuda").to(torch.bfloat16)
            weights = torch.softmax(torch.randn(T, TOP_K, device="cuda"), -1)
            out = torch.empty(T, L, device="cuda")
            ws = torch.empty(lib.lx_moe_decode_workspace_bytes(T, I, TOP_K), dtype=torch.uint8, device="cuda")
            if fmt == "bf16 (lead)":
                args_gu, args_dn = ctypes.c_void_p(gu_ptr), ctypes.c_void_p(dn_ptr)

                def run(i, args_gu=args_gu, args_dn=args_dn):
                    call(kernel, latent, ids_list[i], weights, args_gu, args_dn, out, ws, T, L, I, TOP_K)

            else:
                def run(i, gu_w=gu_w, dn_w=dn_w):
                    call(kernel, latent, ids_list[i], weights, ctypes.pointer(gu_w), ctypes.pointer(dn_w), out, ws,
                         T, L, I, TOP_K)

            mean_us = timed(run, n, flush)
            mean_unique = statistics.mean(uniques)
            rows.append((fmt, T, mean_us, mean_unique, mean_unique * per_expert))
        del keep

    print(f"{'format':12} {'T':>4} {'mean us':>10} {'experts':>8} {'MB moved':>9} {'GB/s':>7} {'% of 896':>9}")
    for name, T, us, uniq, nbytes in rows:
        gbs = nbytes / (us * 1e-6) / 1e9
        print(f"{name:12} {T:>4} {us:>10.2f} {uniq:>8.1f} {nbytes / 1e6:>9.2f} {gbs:>7.0f} {100 * gbs / PEAK_GBS:>8.1f}%")


if __name__ == "__main__":
    main()
