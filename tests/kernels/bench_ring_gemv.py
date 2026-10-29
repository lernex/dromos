"""Program GEMV vs lx_gemv: one product at M = 1, run as a decode program (the persistent kernel's GEMV phase, the one
the decode step uses) and as lx_gemv, on the same weight. Prints GB/s for both and the ratio.

    LX_KERNELS=/path/libdromos_kernels.so python3 tests/kernels/bench_ring_gemv.py [--kinds bf16 int8] [--shapes N,K ...]

Timing is CUDA events around back-to-back runs (no host sync inside the loop); the program's result is checked against a
float64 reference before timing. The bytes counted are the weight's bytes (the same for both paths).
"""

import argparse
import ctypes
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import BF16, F32, Kernels, close  # noqa: E402
from test_decode_program import Program, fx_plain, fy_store, weight, W_INT8  # noqa: E402

DEFAULT_SHAPES = [(16384, 4096), (10304, 2048), (65536, 2048), (2048, 4096)]
FLUSH = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")  # 128 MB: larger than the L2


def bench(lx, N, K, kind, runs):
    lib = lx.lib
    torch.manual_seed(N + K)
    w, wd, keep = weight(N, K, kind)
    x = torch.randn(1, K, device="cuda").to(torch.bfloat16)
    out = torch.zeros(1, N, device="cuda")
    y_lx = torch.zeros(1, N, device="cuda")
    c_stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    prog = Program(lx, 1)
    prog.gemv(fx_plain(x), w, fy_store(out), N, K)
    prog.build()
    lib.lx_program_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    def run_prog():
        status = lib.lx_program_run(prog.p, c_stream)
        assert status == 0, f"lx_program_run returned {status}"

    if kind == "int8":
        w_ptr, w_type, scale = keep[0].data_ptr(), W_INT8, ctypes.c_void_p(keep[1].data_ptr())
    else:
        w_ptr, w_type, scale = keep[0].data_ptr(), 0, ctypes.c_void_p(None)

    def run_lx():
        status = lib.lx_gemv(ctypes.c_void_p(x.data_ptr()), ctypes.c_int(BF16), ctypes.c_int(K),
                             ctypes.c_void_p(w_ptr), ctypes.c_int(w_type), scale,
                             ctypes.c_void_p(None), ctypes.c_int(BF16), ctypes.c_void_p(y_lx.data_ptr()),
                             ctypes.c_int(F32), ctypes.c_int(N), ctypes.c_int(1), ctypes.c_int(N), ctypes.c_int(K),
                             ctypes.c_float(1.0), ctypes.c_float(0.0), c_stream)
        assert status == 0, f"lx_gemv returned {status}"

    # Each timed run starts with the L2 full of something else (a decode step streams weights from DRAM), and the
    # result is the median over the runs.
    def seconds(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        times = []
        for i in range(runs):
            FLUSH.fill_(i & 1)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1e-3)
        times.sort()
        return times[len(times) // 2]

    run_prog()
    torch.cuda.synchronize()
    want = x.double() @ wd.T
    close(out.double(), want, atol=2e-4 * want.abs().max().item() + 1e-30)
    run_lx()
    torch.cuda.synchronize()
    nbytes = N * K * (2 if kind == "bf16" else 1)
    t_prog, t_lx = seconds(run_prog), seconds(run_lx)
    gp, gl = nbytes / t_prog / 1e9, nbytes / t_lx / 1e9
    print(f"{kind:5s} N={N:6d} K={K:5d}  program {t_prog * 1e6:8.1f} us {gp:7.1f} GB/s   "
          f"lx_gemv {t_lx * 1e6:8.1f} us {gl:7.1f} GB/s   program/lx {gp / gl:6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["bf16", "int8"])
    ap.add_argument("--shapes", nargs="+", default=None, help="N,K pairs, e.g. 16384,4096")
    ap.add_argument("--runs", type=int, default=20)
    a = ap.parse_args()
    shapes = [tuple(int(v) for v in s.split(",")) for s in a.shapes] if a.shapes else DEFAULT_SHAPES
    lx = Kernels(os.environ["LX_KERNELS"])
    for kind in a.kinds:
        for N, K in shapes:
            bench(lx, N, K, kind, a.runs)


if __name__ == "__main__":
    main()
