"""How fast the persistent decode kernel streams weights: products through lx_program, timed with CUDA
events, against the card's streaming read rate.

    $PY tools/bench_decode_program.py            (LX_KERNELS points at build/libdromos_kernels.so)

Cases: one in_proj-sized product; a chain of them where each reads the one before (every op waits on
the previous one, so this measures how well the ring hides the dependency); the same products with
no dependencies; and a Metis-like layer mix of small and large products in a chain.
"""

import ctypes
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "kernels"))
from conftest import Kernels  # noqa: E402
from test_decode_program import Program, fx_plain, fy_store, weight  # noqa: E402


def timed(prog, iters=50):
    prog.run()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    start.record()
    for _ in range(iters):
        assert prog.lib.lx_program_run(prog.p, stream) == 0
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e-3


def report(name, seconds, nbytes):
    print(f"{name:44s} {seconds * 1e6:9.1f} us  {nbytes / 1e6:8.1f} MB  {nbytes / seconds / 1e9:6.1f} GB/s")


def copy_rate():
    a = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    b = torch.empty_like(a)
    b.copy_(a)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        b.copy_(a)
    end.record()
    torch.cuda.synchronize()
    return 2 * a.numel() * 10 / (start.elapsed_time(end) * 1e-3)


def chain(lx, M, shapes, kind, dependent=True):
    prog = Program(lx, M)
    keep, total = [], 0
    x = torch.randn(M, shapes[0][1], device="cuda").to(torch.bfloat16)
    keep.append(x)
    for N, K in shapes:
        w, _, k = weight(N, K, kind)
        keep += k
        out = torch.zeros(M, N, device="cuda").to(torch.bfloat16)
        src = x if dependent or len(keep) == 1 else keep[0]
        if src.shape[1] != K:
            src = torch.randn(M, K, device="cuda").to(torch.bfloat16)
            keep.append(src)
        prog.gemv(fx_plain(src), w, fy_store(out), N, K)
        keep.append(out)
        if dependent:
            x = out
        total += N * K * (1 if kind == "int8" else 2) + (N * 4 if kind == "int8" else 0)
    prog.build()
    prog.keep += keep
    return prog, total


def main():
    lx = Kernels(os.environ["LX_KERNELS"])
    torch.manual_seed(0)
    print(f"device copy: {copy_rate() / 1e9:.1f} GB/s (read + write)")
    for kind in ["int8", "bf16"]:
        for M in [1, 8]:
            prog, nbytes = chain(lx, M, [(10304, 2048)], kind)
            report(f"{kind} M={M} one 10304x2048", timed(prog), nbytes)
            prog, nbytes = chain(lx, M, [(10304, 2048)] * 30, kind, dependent=False)
            report(f"{kind} M={M} 30 x 10304x2048, independent", timed(prog, 10), nbytes)
            shapes = [(2048, 2048)] * 60
            prog, nbytes = chain(lx, M, shapes, kind)
            report(f"{kind} M={M} 60 x 2048x2048, chained", timed(prog, 10), nbytes)
            # one Mamba layer's products, chained: in_proj, out_proj, latent_down, latent_up, x 12
            layer = [(10304, 2048), (2048, 10304), (1024, 2048), (2048, 1024)]
            prog, nbytes = chain(lx, M, layer * 12, kind)
            report(f"{kind} M={M} layer mix x12, chained", timed(prog, 10), nbytes)


if __name__ == "__main__":
    main()
