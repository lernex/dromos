"""The Mamba-2 step of a decode step, per layer-pass, against its state-bytes floor, at 1 to 64 sequences.

Run on the GPU server, after a build so build/liblernex_kernels.so is current:
    scripts/remote.sh mamba2 build
    scripts/remote.sh mamba2 x '$PY bench/bench_mamba_step.py'

Two paths, the same arithmetic and the same bits: "split" is lx_conv1d_varlen, lx_ssm_scan_varlen and
lx_gated_rmsnorm in turn (the kernel-per-op path); "fused" is lx_mamba_decode_step, one kernel. Each graph
replay makes CALLS layer-passes, cycling through CYCLE distinct state buffers, so every call streams its
state from DRAM (CYCLE x the state, about 87 MB at one sequence, is more than the L2). The floor is the
bytes a step must move: each sequence's conv and SSM state read once and written once, at the card's
peak bandwidth. Times are medians over repeated replays, in microseconds per layer-pass.
"""

import ctypes
import os
import statistics

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.environ.get("LX_KERNELS", os.path.join(ROOT, "build", "liblernex_kernels.so"))
_lib = ctypes.CDLL(LIB)

# The Mamba-2 shape of a layer: Praxis by default, LX_MAMBA_SHAPE=logos for Logos (80 heads in 10 groups)
SHAPES = {"praxis": (64, 64, 8, 128, 4), "logos": (80, 64, 10, 128, 4)}
SHAPE_NAME = os.environ.get("LX_MAMBA_SHAPE", "praxis")
H, P, G, N, W = SHAPES[SHAPE_NAME]
I = H * P
GN = G * N
CONV_DIM = I + 2 * GN
PROJ = 2 * I + 2 * GN + H
EPS = 1e-5
PEAK_BYTES_PER_S = 896e9  # RTX 5070 Ti
STREAMS = (1, 2, 4, 8, 16, 32, 64)
CALLS = 40  # layer-passes per graph replay
CYCLE = 40  # distinct state buffers they cycle through
REPEATS = 9


def call(name, *args):
    """Launch lx_<name> on the current stream without synchronizing, so it can be captured."""
    fn = getattr(_lib, "lx_" + name)
    fn.restype = ctypes.c_int
    conv = []
    for a in args:
        if isinstance(a, torch.Tensor):
            conv.append(ctypes.c_void_p(a.data_ptr()))
        elif isinstance(a, ctypes.c_void_p):
            conv.append(a)
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


def at(t, elems):
    return ctypes.c_void_p(t.data_ptr() + elems * t.element_size())


def graph_us_per_call(fn):
    """Median device time per layer-pass: CALLS layer-passes captured in one graph, replayed."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
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
        times.append(start.elapsed_time(end) * 1000.0 / CALLS)
    return statistics.median(times)


def setup(n):
    """Inputs for n sequences; the state buffers are CYCLE layer-passes' worth."""
    torch.manual_seed(0)
    dev = "cuda"
    ld = PROJ
    zb = torch.randn(n, ld, device=dev)
    w = dict(
        conv_w=(torch.randn(CONV_DIM, W, device=dev) * 0.5).to(torch.bfloat16),
        conv_b=(torch.randn(CONV_DIM, device=dev) * 0.1).to(torch.bfloat16),
        A_log=torch.log(torch.rand(H, device=dev) * 3.5 + 0.5),
        D=torch.randn(H, device=dev),
        dt_bias=torch.randn(H, device=dev) * 0.5,
        norm_w=(torch.rand(I, device=dev) + 0.5).to(torch.bfloat16),
    )
    slots = torch.arange(n, dtype=torch.int32, device=dev)
    has = torch.ones(n, dtype=torch.int32, device=dev)
    conv = [torch.randn(n, W - 1, CONV_DIM, device=dev) for _ in range(CYCLE)]
    ssm = [torch.randn(n, H, P, N, device=dev) * 0.5 for _ in range(CYCLE)]
    xbc = torch.empty(n, CONV_DIM, device=dev)
    y = torch.empty(n, I, device=dev)
    out = torch.empty(n, I, device=dev, dtype=torch.bfloat16)
    sync = torch.zeros(n * G, dtype=torch.int32, device=dev)
    cu = torch.arange(n + 1, dtype=torch.int32, device=dev)
    return dict(zb=zb, w=w, slots=slots, has=has, conv=conv, ssm=ssm, xbc=xbc, y=y, out=out, sync=sync, cu=cu,
                n=n, ld=ld)


def split_call(d, k):
    w, n, ld = d["w"], d["n"], d["ld"]
    zb, xbc, y, out = d["zb"], d["xbc"], d["y"], d["out"]
    call("conv1d_varlen", at(zb, I), ld, w["conv_w"], w["conv_b"], d["conv"][k], d["slots"], d["has"], d["cu"], n,
         xbc, CONV_DIM, W)
    call("ssm_scan_varlen", xbc, CONV_DIM, at(zb, I + CONV_DIM), ld, at(xbc, I), at(xbc, I + GN), CONV_DIM,
         w["A_log"], w["D"], w["dt_bias"], d["ssm"][k], d["slots"], d["has"], d["cu"], n, y, H, P, G, N)
    call("gated_rmsnorm", y, zb, ld, w["norm_w"], out, n, I, G, EPS)


def fused_call(d, k):
    w, n, ld = d["w"], d["n"], d["ld"]
    call("mamba_decode_step", d["zb"], ld, w["conv_w"], w["conv_b"], d["conv"][k], d["ssm"][k], w["A_log"], w["D"],
         w["dt_bias"], d["slots"], d["has"], w["norm_w"], EPS, d["y"], d["out"], d["sync"], n, H, P, G, N, W)


def run(path, d):
    fn = split_call if path == "split" else fused_call

    def layer_passes():
        for c in range(CALLS):
            fn(d, c % CYCLE)

    return graph_us_per_call(layer_passes)


def main():
    state_bytes = 4 * (H * P * N + (W - 1) * CONV_DIM)  # one sequence, one layer-pass
    print(f"Mamba-2 step, {SHAPE_NAME} shape: H={H} P={P} G={G} N={N} W={W}, state {state_bytes / 2**20:.2f} MiB per "
          f"sequence and layer-pass (read + write = {2 * state_bytes / 2**20:.2f} MiB)")
    print(f"{'streams':>8} {'split us':>10} {'fused us':>10} {'floor us':>10} {'fused/floor':>12} {'fused GB/s':>11}")
    for n in STREAMS:
        d = setup(n)
        t_split = run("split", d)
        t_fused = run("fused", d)
        floor = 2 * state_bytes * n / PEAK_BYTES_PER_S * 1e6
        gbs = 2 * state_bytes * n / (t_fused * 1e-6) / 1e9
        print(f"{n:>8} {t_split:>10.2f} {t_fused:>10.2f} {floor:>10.2f} {t_fused / floor:>12.2f} {gbs:>11.0f}",
              flush=True)
        del d
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
