"""Decode GEMV benchmark: lx_gemv (bf16 and int8 weights) against DRAM bandwidth, and lx_gemm_bf16.

Every launch reads its own weight copy, and the copies together are several times the 48 MB L2, so
the reads come from DRAM. The launches are captured in one CUDA graph and replayed, so the time is
the GPU's rather than the Python call's. Each per-launch time includes the gap between graph nodes;
the floor row times a trivial kernel in the same kind of graph to show that gap.

Run: scripts/remote.sh gemv run '$PY bench/bench_gemv.py'
"""

import ctypes
import os
import statistics

import torch

F32, BF16 = 0, 1
W_BF16, W_INT8 = 0, 1
DRAM_GBPS = 896.0  # RTX 5070 Ti: 256-bit GDDR7 at 28 Gb/s
L2_BYTES = 48 << 20
ROTATE_BYTES = 192 << 20  # copies rotated per graph; well above L2 so no read hits it

lib = ctypes.CDLL(os.environ["LX_KERNELS"])
c = ctypes
lib.lx_gemv.restype = c.c_int
lib.lx_gemv.argtypes = [
    c.c_void_p, c.c_int, c.c_int, c.c_void_p, c.c_int, c.c_void_p, c.c_void_p, c.c_int,
    c.c_void_p, c.c_int, c.c_int, c.c_int, c.c_int, c.c_int, c.c_float, c.c_float, c.c_void_p,
]
lib.lx_gemm_bf16.restype = c.c_int
lib.lx_gemm_bf16.argtypes = [
    c.c_void_p, c.c_int, c.c_void_p, c.c_void_p, c.c_void_p, c.c_int, c.c_int, c.c_int, c.c_int, c.c_int,
    c.c_float, c.c_float, c.c_void_p,
]
lib.lx_activate_f32.restype = c.c_int
lib.lx_activate_f32.argtypes = [c.c_void_p, c.c_int64, c.c_int, c.c_void_p]


def ok(status):
    assert status == 0, f"CUDA error {status}"


def stream():
    return torch.cuda.current_stream().cuda_stream


def make_copies(N, K, w_type, copies):
    """Independent weight copies: (w, scale) with scale None for bf16."""
    out = []
    for _ in range(copies):
        if w_type == "bf16":
            out.append((torch.empty(N, K, dtype=torch.bfloat16, device="cuda").normal_(), None))
        else:
            q = torch.empty(N, K, dtype=torch.int8, device="cuda").random_(-127, 128)
            scale = torch.rand(N, device="cuda") * 0.01 + 0.001
            out.append((q, scale))
    return out


def graph_us_per_launch(launches, replays=10, reps=5):
    """Median time per launch, from CUDA events around graph replays of all the launches."""
    for f in launches:  # warm up: loads kernels, builds cuBLASLt plans
        f()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for f in launches:
            f()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / (replays * len(launches)))
    return statistics.median(times)


def copies_for(bytes_per_copy):
    return max(3, min(128, -(-ROTATE_BYTES // bytes_per_copy)))


def gemv_row(N, K, M, w_type, x_dtype=BF16):
    bytes_per_copy = N * K * (2 if w_type == "bf16" else 1) + (4 * N if w_type == "int8" else 0)
    copies = make_copies(N, K, w_type, copies_for(bytes_per_copy))
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16 if x_dtype == BF16 else torch.float32)
    y = torch.empty(M, N, dtype=torch.float32, device="cuda")
    w_code = W_BF16 if w_type == "bf16" else W_INT8

    def make(w, scale):
        return lambda: ok(
            lib.lx_gemv(
                x.data_ptr(), x_dtype, K, w.data_ptr(), w_code, scale.data_ptr() if scale is not None else None,
                None, F32, y.data_ptr(), F32, N, M, N, K, 1.0, 0.0, stream(),
            )
        )

    us = graph_us_per_launch([make(w, s) for w, s in copies])
    del copies
    return us, bytes_per_copy


def gemm_row(N, K, M):
    bytes_per_copy = N * K * 2
    copies = make_copies(N, K, "bf16", copies_for(bytes_per_copy))
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    y = torch.empty(M, N, dtype=torch.float32, device="cuda")

    def make(w):
        return lambda: ok(
            lib.lx_gemm_bf16(x.data_ptr(), K, w.data_ptr(), None, y.data_ptr(), F32, N, M, N, K, 1.0, 0.0, stream())
        )

    us = graph_us_per_launch([make(w) for w, _ in copies])
    del copies
    return us


def floor_us():
    """A trivial kernel in the same kind of graph: the per-node cost the rows above include."""
    buf = torch.zeros(1024, device="cuda")
    return graph_us_per_launch([lambda: ok(lib.lx_activate_f32(buf.data_ptr(), 1, 2, stream()))] * 64)


BIG = [(10304, 2048), (3072, 2048), (2048, 4096), (65536, 2048), (1024, 2048), (2048, 1024)]
SMALL = [(256, 6400), (256, 2048), (256, 256), (24, 64), (8, 2048), (3072, 8), (128, 1280), (1, 256)]


def gbps(bytes_moved, us):
    return bytes_moved / (us * 1e-6) / 1e9


def main():
    print(f"device: {torch.cuda.get_device_name(0)}; DRAM {DRAM_GBPS:.0f} GB/s; rotation {ROTATE_BYTES >> 20} MB")
    print(f"floor (trivial kernel per graph node): {floor_us():.2f} us")
    print()
    print("| N x K | weights | M | x | us | GB/s | % of 896 | cuBLASLt bf16 us | GEMV / cuBLASLt |")
    print("|---|---|---|---|---|---|---|---|---|")
    for N, K in BIG + SMALL:
        is_big = (N, K) in BIG
        for w_type in ["bf16", "int8"]:
            for M in [1, 8]:
                us, nbytes = gemv_row(N, K, M, w_type)
                gb = gbps(nbytes, us)
                cub = ratio = "-"
                if w_type == "bf16" and is_big:
                    cub_us = gemm_row(N, K, M)
                    cub = f"{cub_us:.2f}"
                    ratio = f"{us / cub_us:.2f}"
                print(
                    f"| {N} x {K} | {w_type} | {M} | bf16 | {us:.2f} | {gb:.0f} | {100 * gb / DRAM_GBPS:.1f} | "
                    f"{cub} | {ratio} |",
                    flush=True,
                )
        if is_big:
            us, nbytes = gemv_row(N, K, 1, "bf16", x_dtype=F32)
            print(
                f"| {N} x {K} | bf16 | 1 | f32 | {us:.2f} | {gbps(nbytes, us):.0f} | "
                f"{100 * gbps(nbytes, us) / DRAM_GBPS:.1f} | - | - |",
                flush=True,
            )


if __name__ == "__main__":
    main()
