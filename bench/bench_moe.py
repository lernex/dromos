#!/usr/bin/env python3
"""Single-call timing of the expert kernels at Metis shapes (E=128, L=1024, I=512, top_k=4).

Every timed call reads its expert tables from DRAM. A ~256 MB float read runs before each call and
evicts the 64 MB L2, and each call gets its own random routing, drawn ahead of time. CUDA events
bracket each call alone. Reported: the trimmed mean of those per-call times (the fastest and slowest
tenth are dropped), the bytes the distinct experts of a call move, and that as a share of the 896 GB/s
the card is rated for.

Environment: LX_KERNELS (the built library, set by scripts/remote.sh), N (calls per shape, default 200;
int8 shapes with T >= 64 use a quarter of that, at least 40), W8_ONLY (skip the bf16 baseline rows,
for variant builds without lx_moe_experts).

Run on the GPU server: scripts/remote.sh experts run '$PY bench/bench_moe.py'
"""

import ctypes
import os
import random
import statistics
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "kernels"))
from test_moe_w8 import quantize_w8, workspace_bytes  # noqa: E402

E, L, I, TOP_K = 128, 1024, 512, 4
PEAK_GBS = 896.0

TOKENS = [1, 2, 4, 8, 64, 300]
BF16_TOKENS = [1, 8, 64, 300]  # the bf16 kernel, for the baseline it replaces
FLUSH_BYTES = 256 << 20


def call(fn, *args):
    conv = [ctypes.c_void_p(a.data_ptr()) if isinstance(a, torch.Tensor) else ctypes.c_int(a) for a in args]
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


def main():
    lib = ctypes.CDLL(os.environ["LX_KERNELS"])
    w8 = lib.lx_moe_experts_w8
    w8.restype = ctypes.c_int
    only_w8 = bool(os.environ.get("W8_ONLY"))  # variant builds of the w8 file have no bf16 kernel
    bf = None if only_w8 else lib.lx_moe_experts
    if bf is not None:
        bf.restype = ctypes.c_int
    n_iter = int(os.environ.get("N", "200"))
    shim = types.SimpleNamespace(lib=lib)

    torch.manual_seed(0)
    # Tables are shared by all T: 128 experts, 192 MB int8 (or 384 MB bf16), far larger than L2.
    gate_up = (torch.randn(E, 2 * I, L, device="cuda") / L**0.5).to(torch.bfloat16)
    down = (torch.randn(E, L, I, device="cuda") / I**0.5).to(torch.bfloat16)
    q_gu, s_gu = quantize_w8(gate_up)
    q_dn, s_dn = quantize_w8(down)
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")

    rows = []
    for T in TOKENS:
        n = n_iter if T < 64 else max(40, n_iter // 4)
        ids_list, uniques = routing(T, n)
        latent = torch.randn(T, L, device="cuda").to(torch.bfloat16)
        weights = torch.softmax(torch.randn(T, TOP_K, device="cuda"), -1)
        out = torch.empty(T, L, device="cuda")
        ws = torch.empty(workspace_bytes(shim, T, I, TOP_K), dtype=torch.uint8, device="cuda")

        def run(i):
            call(w8, latent, ids_list[i], weights, q_gu, s_gu, q_dn, s_dn, out, ws, T, L, I, TOP_K)

        mean_us = timed(run, n, flush)
        # Bytes the kernel must move: int8 weights and f32 scales of each distinct expert, once.
        per_expert = (2 * I * L + L * I) + 4 * (2 * I + L)
        mean_unique = statistics.mean(uniques)
        rows.append(("int8 w8", T, mean_us, mean_unique, mean_unique * per_expert))

    for T in ([] if only_w8 else BF16_TOKENS):
        n = n_iter
        ids_list, uniques = routing(T, n)
        latent = torch.randn(T, L, device="cuda").to(torch.bfloat16)
        weights = torch.softmax(torch.randn(T, TOP_K, device="cuda"), -1)
        out = torch.empty(T, L, device="cuda")

        def run_bf(i):
            call(bf, latent, ids_list[i], weights, gate_up, down, out, T, L, I, TOP_K)

        mean_us = timed(run_bf, n, flush)
        per_expert = 2 * ((2 * I * L) + (L * I))
        mean_unique = statistics.mean(uniques)
        rows.append(("bf16", T, mean_us, mean_unique, mean_unique * per_expert))

    print(f"{'kernel':8} {'T':>4} {'mean us':>10} {'experts':>8} {'MB moved':>9} {'GB/s':>7} {'% of 896':>9}")
    for name, T, us, uniq, nbytes in rows:
        gbs = nbytes / (us * 1e-6) / 1e9
        print(f"{name:8} {T:>4} {us:>10.2f} {uniq:>8.1f} {nbytes / 1e6:>9.2f} {gbs:>7.0f} {100 * gbs / PEAK_GBS:>8.1f}%")


if __name__ == "__main__":
    main()
