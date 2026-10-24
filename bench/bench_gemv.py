"""Decode GEMV benchmark: lx_gemv_w on every weight format against DRAM bandwidth.

Every launch reads its own weight copy, and the copies together are several times the 48 MB L2, so the
reads come from DRAM. The launches are captured in one CUDA graph and replayed, so the time is the GPU's
rather than the Python call's. Each per-launch time includes the gap between graph nodes; the floor row
times a trivial kernel in the same kind of graph to show that gap.

Bytes moved per launch are the weight bytes and the scales the format reads (bf16: 2 per value; int8 and
fp8: 1 per value and an f32 per row; nvfp4: 0.5 per value, an e4m3 per 16 values, and the global).

Environment: LX_KERNELS (set by scripts/remote.sh). FORMATS (comma list, default all), M (default 1,2,4,8),
SHAPES (N:K pairs, default the Praxis set below).
Run on the GPU server: scripts/remote.sh formats x '$PY bench/bench_gemv.py'
"""

import ctypes
import os
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "kernels"))
from conftest import DTYPE  # noqa: E402
from formats import FORMATS, LxWeight  # noqa: E402

DRAM_GBPS = 896.0  # RTX 5070 Ti: 256-bit GDDR7 at 28 Gb/s
ROTATE_BYTES = 192 << 20  # copies rotated per graph; well above L2 so no read hits it
F32, BF16 = 0, 1

lib = ctypes.CDLL(os.environ["LX_KERNELS"])
lib.lx_gemv_w.restype = ctypes.c_int
lib.lx_gemv_w.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ctypes.c_float, ctypes.c_void_p,
]
lib.lx_activate_f32.restype = ctypes.c_int
lib.lx_activate_f32.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p]


def ok(status):
    assert status == 0, f"CUDA error {status}"


def stream():
    return torch.cuda.current_stream().cuda_stream


def weight_bytes(N, K, fmt):
    if fmt == "bf16":
        return 2 * N * K
    if fmt in ("int8", "fp8"):
        return N * K + 4 * N
    return N * K // 2 + N * K // 16  # nvfp4 data and e4m3 block scales (the global is one float)


def make_copy(N, K, fmt):
    """One weight in format `fmt`, random bytes (the timing does not depend on the values)."""
    if fmt == "bf16":
        data = torch.empty(N, K, dtype=torch.bfloat16, device="cuda").normal_()
        return LxWeight(0, data.data_ptr(), None, 1.0, None), (data,)
    if fmt in ("int8", "fp8"):
        data = torch.randint(0, 256, (N, K), dtype=torch.uint8, device="cuda")
        scale = torch.rand(N, device="cuda") * 0.01 + 0.001
        code = 1 if fmt == "int8" else 2
        return LxWeight(code, data.data_ptr(), scale.data_ptr(), 1.0, None), (data, scale)
    data = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    scale = torch.randint(0, 120, (N, K // 16), dtype=torch.uint8, device="cuda")  # small positive e4m3
    return LxWeight(3, data.data_ptr(), scale.data_ptr(), 0.05, None), (data, scale)


def graph_us_per_launch(launches, replays=10, reps=5):
    """Median time per launch, from CUDA events around graph replays of all the launches."""
    for f in launches:  # warm up: loads kernels, builds the tuning cache
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


def gemv_row(N, K, M, fmt, x_dtype=BF16):
    nbytes = weight_bytes(N, K, fmt)
    copies = []
    for _ in range(copies_for(nbytes)):
        copies.append(make_copy(N, K, fmt))
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16 if x_dtype == BF16 else torch.float32)
    y = torch.empty(M, N, dtype=torch.float32, device="cuda")

    def make(lw, keep):
        pointer = ctypes.pointer(lw)
        return lambda: ok(
            lib.lx_gemv_w(
                x.data_ptr(), x_dtype, K, pointer, None, F32, y.data_ptr(), F32, N, M, N, K, 1.0, 0.0, stream(),
            )
        )

    us = graph_us_per_launch([make(lw, keep) for lw, keep in copies])
    del copies
    return us, nbytes


def floor_us():
    """A trivial kernel in the same kind of graph: the per-node cost the rows above include."""
    buf = torch.zeros(1024, device="cuda")
    return graph_us_per_launch([lambda: ok(lib.lx_activate_f32(buf.data_ptr(), 1, 2, stream()))] * 64)


# The Praxis decode products (N x K), and the logits projection.
SHAPES = [(10304, 2048), (2048, 4096), (3072, 2048), (2048, 2048), (1024, 2048), (2048, 1024), (65536, 2048)]


def main():
    formats = os.environ.get("FORMATS", ",".join(FORMATS)).split(",")
    ms = [int(m) for m in os.environ.get("M", "1,2,4,8").split(",")]
    shapes = SHAPES
    if "SHAPES" in os.environ:
        shapes = [tuple(int(v) for v in s.split(":")) for s in os.environ["SHAPES"].split(",")]
    print(f"device: {torch.cuda.get_device_name(0)}; DRAM {DRAM_GBPS:.0f} GB/s; rotation {ROTATE_BYTES >> 20} MB")
    print(f"floor (trivial kernel per graph node): {floor_us():.2f} us")
    print()
    print("| N x K | format | M | us | GB/s | % of 896 |")
    print("|---|---|---|---|---|---|")
    for N, K in shapes:
        for fmt in formats:
            for M in ms:
                us, nbytes = gemv_row(N, K, M, fmt)
                gb = nbytes / (us * 1e-6) / 1e9
                print(f"| {N} x {K} | {fmt} | {M} | {us:.2f} | {gb:.0f} | {100 * gb / DRAM_GBPS:.1f} |", flush=True)


if __name__ == "__main__":
    main()
