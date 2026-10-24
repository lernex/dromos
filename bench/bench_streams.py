"""Per-call time of the residual-stream and depth-memory kernels at decode and prefill widths.

Run on the GPU server, after a build so build/liblernex_kernels.so is current:
    scripts/remote.sh streams2 build
    scripts/remote.sh streams2 run '$PY bench/bench_streams.py'

Two numbers per row:
  graph   the call captured 200 times into one CUDA graph and replayed; CUDA events bracket the
          replays. This is the kernel plus its launch gap, with no host overhead in the way.
  kernel  the device time of the kernel alone, from the profiler (CUPTI), per call.
Each graph figure is the median of repeated replays.
"""

import ctypes
import os
import statistics

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.environ.get("LX_KERNELS", os.path.join(ROOT, "build", "liblernex_kernels.so"))
_lib = ctypes.CDLL(LIB)

BF16 = 1
S = 4
M = 256
E = 8
EPS = 1e-6
T_VALUES = (1, 4, 8, 300)
D_VALUES = (2048, 2560)  # Praxis, Logos
GRAPH_CALLS = 200
REPEATS = 15


def call(name, *args):
    """Launch lx_<name> on the current stream without synchronizing, so it can be captured."""
    fn = getattr(_lib, "lx_" + name)
    fn.restype = ctypes.c_int
    conv = []
    for a in args:
        if isinstance(a, torch.Tensor):
            conv.append(ctypes.c_void_p(a.data_ptr()))
        elif isinstance(a, bool):
            conv.append(ctypes.c_int(int(a)))
        elif isinstance(a, int):
            conv.append(ctypes.c_int(a))
        elif isinstance(a, float):
            conv.append(ctypes.c_float(a))
        elif a is None:
            conv.append(ctypes.c_void_p(None))
        else:
            raise TypeError(type(a))
    conv.append(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    status = fn(*conv)
    if status != 0:
        raise RuntimeError(f"lx_{name} returned CUDA error {status}")


def graph_us(fn):
    """Median device-side time per call, in microseconds, of GRAPH_CALLS calls replayed as a graph."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(GRAPH_CALLS):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    times = []
    for _ in range(REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / GRAPH_CALLS)
    return statistics.median(times)


def kernel_us(fn, needle, calls=50):
    """Device time per call of the kernels whose name contains needle, from the profiler."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    total = 0.0
    for ev in prof.events():
        if needle in ev.name and ev.device_type == torch.autograd.DeviceType.CUDA:
            total += ev.device_time
    return total / calls


def cases(T, D):
    """(name, needle, callable) for each kernel at this width. Buffers are allocated once per case."""
    dev = "cuda"
    f32 = dict(device=dev, dtype=torch.float32)
    bf16 = dict(device=dev, dtype=torch.bfloat16)
    streams = torch.randn(T, S, D, **f32)
    mixed = torch.randn(T, S, D, **f32)
    out_streams = torch.empty(T, S, D, **f32)
    mean = torch.empty(T, D, **f32)
    matrix = torch.randn(S, S, **f32)
    read = torch.randn(S, **f32)
    norm_w = torch.rand(D, **bf16) + 0.5
    h = torch.empty(T, D, **bf16)
    write = torch.randn(S, **f32)
    update = torch.randn(T, D, **f32)
    projected = torch.randn(T, S, D, **f32) * 0.5
    gate = torch.randn(S, D, **bf16)
    gate_bias = torch.randn(S, **bf16)
    summary = torch.empty(T, D, **f32)
    q = torch.randn(T, S, M, **f32)
    bank_k = torch.randn(T, E, M, **f32)
    bank_v = torch.randn(T, E, M, **f32)
    attn_out = torch.empty(T, S, M, **bf16)

    return [
        ("stream_mean", "stream_mean", lambda: call("stream_mean", streams, mean, T, S, D)),
        ("mhc_read", "mhc_read", lambda: call("mhc_read", streams, matrix, read, norm_w, EPS, mixed, h, T, S, D)),
        ("mhc_write", "mhc_write", lambda: call("mhc_write", mixed, write, update, out_streams, T, S, D)),
        ("memory_attend", "memory_attend",
         lambda: call("memory_attend", q, bank_k, bank_v, E, E, attn_out, T, S, M)),
        ("memory_fuse", "memory_fuse",
         lambda: call("memory_fuse", streams, projected, gate, gate_bias, summary, T, S, D)),
    ]


def main():
    print(f"device {torch.cuda.get_device_name(0)}  S={S} M={M} entries={E}  lib={LIB}")
    print(f"{'kernel':<14}{'D':>6}{'T':>5}{'graph us':>11}{'kernel us':>11}")
    for D in D_VALUES:
        for T in T_VALUES:
            for name, needle, fn in cases(T, D):
                g = graph_us(fn)
                k = kernel_us(fn, needle)
                print(f"{name:<14}{D:>6}{T:>5}{g:>11.2f}{k:>11.2f}", flush=True)


if __name__ == "__main__":
    main()
