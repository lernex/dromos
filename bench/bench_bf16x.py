"""Lossless BF16 (LX_W_BF16X) decode benchmark: lx_gemv_bf16x against lx_gemv on the same bf16 weights,
and lx_moe_experts_bf16x against the bf16 expert kernels, at every Praxis decode shape.

GEMV: every launch reads its own weight copy (Gaussian, so patches occur at Praxis's rate), and the
copies together are several times the 48 MB L2, so reads come from DRAM. The launches are captured in
one CUDA graph and replayed (programmatic dependent launch between them, as in a decode step).
GB/s is on the bytes each format actually moves: bf16 2 bytes a weight, BF16X its encoded size.

Experts (E = 128, L = 1024, I = 512, top_k = 4): eight independent expert stacks, used in turn by
the graph's calls, each with its own random routing, so no call finds its experts in L2. Bytes are
the distinct experts' gate_up and down rows in the kernel's format. Baselines: lx_moe_experts (moe.cu,
what decode runs on bf16 today) and lx_moe_experts_decode_bf16 (this file's kernels on plain bf16, the
like-for-like comparison).

Run: scripts/remote.sh SLOT x '$PY bench/bench_bf16x.py [--sweep]'
  --sweep  also time every GEMV variant per shape (rows per warp, steps in flight)
"""

import ctypes
import os
import statistics
import sys

import torch

F32, BF16 = 0, 1
W_BF16 = 0
DRAM_GBPS = 896.0
ROTATE_BYTES = 192 << 20

lib = ctypes.CDLL(os.environ["LX_KERNELS"])
c = ctypes
lib.lx_gemv.restype = c.c_int
lib.lx_gemv.argtypes = [
    c.c_void_p, c.c_int, c.c_int, c.c_void_p, c.c_int, c.c_void_p, c.c_void_p, c.c_int,
    c.c_void_p, c.c_int, c.c_int, c.c_int, c.c_int, c.c_int, c.c_float, c.c_float, c.c_void_p,
]
lib.lx_gemv_bf16x.restype = c.c_int
lib.lx_gemv_bf16x.argtypes = [
    c.c_void_p, c.c_int, c.c_int, c.c_void_p, c.c_void_p, c.c_int,
    c.c_void_p, c.c_int, c.c_int, c.c_int, c.c_int, c.c_int, c.c_float, c.c_float, c.c_void_p,
]
lib.lx_bf16x_bytes.restype = c.c_int64
lib.lx_bf16x_bytes.argtypes = [c.c_void_p, c.c_int64, c.c_int, c.c_void_p]
lib.lx_bf16x_encode.argtypes = [c.c_void_p, c.c_int64, c.c_int, c.c_void_p, c.c_int64, c.c_void_p]
lib.lx_tune_set.argtypes = [c.c_char_p, c.c_int, c.c_int, c.c_int, c.c_int]
lib.lx_tune_clear.argtypes = []
lib.lx_tune_variants.argtypes = [c.c_char_p]
lib.lx_tune_variant_name.restype = c.c_char_p
lib.lx_tune_variant_name.argtypes = [c.c_char_p, c.c_int]
for name in ("lx_moe_experts_bf16x", "lx_moe_experts_decode_bf16"):
    getattr(lib, name).restype = c.c_int
    getattr(lib, name).argtypes = [c.c_void_p] * 7 + [c.c_int] * 4 + [c.c_void_p]
lib.lx_moe_experts.restype = c.c_int
lib.lx_moe_experts.argtypes = [c.c_void_p] * 6 + [c.c_int] * 4 + [c.c_void_p]
lib.lx_moe_bf16x_workspace_bytes.restype = c.c_int64


def ok(status):
    assert status == 0, f"CUDA error {status}"


def stream():
    return torch.cuda.current_stream().cuda_stream


def encode(w):
    rows, K = w.numel() // w.shape[-1], w.shape[-1]
    n = lib.lx_bf16x_bytes(w.data_ptr(), rows, K, stream())
    assert n > 0
    out = torch.empty(n, dtype=torch.uint8, device="cuda")
    ok(lib.lx_bf16x_encode(w.data_ptr(), rows, K, out.data_ptr(), n, stream()))
    return out


def graph_us_per_launch(launches, replays=10, reps=5):
    for f in launches:
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


def gbps(nbytes, us):
    return nbytes / (us * 1e-6) / 1e9


class GemvShape:
    """Weight copies of one shape in both formats, and launchers for each."""

    def __init__(self, N, K):
        self.N, self.K = N, K
        copies = max(3, min(96, -(-ROTATE_BYTES // (int(N * K * 1.52)))))
        self.w = [(torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16) for _ in range(copies)]
        self.x16 = [encode(w) for w in self.w]
        self.bytes_bf16 = 2 * N * K
        self.bytes_x = self.x16[0].numel()

    def time(self, fmt, M, x_dtype=BF16):
        N, K = self.N, self.K
        x = torch.randn(M, K, device="cuda").to(torch.bfloat16 if x_dtype == BF16 else torch.float32)
        y = torch.empty(M, N, device="cuda")
        if fmt == "bf16":
            fs = [lambda w=w: ok(lib.lx_gemv(x.data_ptr(), x_dtype, K, w.data_ptr(), W_BF16, None, None, F32,
                                             y.data_ptr(), F32, N, M, N, K, 1.0, 0.0, stream())) for w in self.w]
        else:
            fs = [lambda w=w: ok(lib.lx_gemv_bf16x(x.data_ptr(), x_dtype, K, w.data_ptr(), None, F32, y.data_ptr(),
                                                   F32, N, M, N, K, 1.0, 0.0, stream())) for w in self.x16]
        return graph_us_per_launch(fs)


SHAPES = [  # (N, K, what) of every Praxis decode GEMV
    (10304, 2048, "mamba in_proj"),
    (2048, 4096, "mamba out_proj"),
    (3072, 2048, "attention qkv"),
    (2048, 2048, "attention out"),
    (1024, 2048, "moe latent_down"),
    (2048, 1024, "moe latent_up"),
    (1024, 1024, "shared gate_up"),
    (1024, 512, "shared down"),
    (65536, 2048, "lm head"),
    (256, 2048, "memory state_write, query"),
    (2048, 256, "memory output"),
    (256, 6400, "memory route_projection"),
]


def gemv_table(sweep):
    name = b"gemv_bf16x"
    nvar = lib.lx_tune_variants(name)
    print("| N x K | product | M | x | bf16 us | bf16 GB/s | bf16 % | BF16X us | BF16X GB/s | BF16X % | speedup |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for N, K, what in SHAPES:
        shape = GemvShape(N, K)
        for M, x_dtype in ((1, BF16), (1, F32), (8, BF16)):
            lib.lx_tune_clear()
            t16 = shape.time("bf16", M, x_dtype)
            tx = shape.time("bf16x", M, x_dtype)
            g16, gx = gbps(shape.bytes_bf16, t16), gbps(shape.bytes_x, tx)
            print(f"| {N} x {K} | {what} | {M} | {'bf16' if x_dtype == BF16 else 'f32'} | {t16:.2f} | {g16:.0f} | "
                  f"{100 * g16 / DRAM_GBPS:.1f} | {tx:.2f} | {gx:.0f} | {100 * gx / DRAM_GBPS:.1f} | {t16 / tx:.3f}x |",
                  flush=True)
            if sweep:
                row = []
                for v in range(nvar):
                    lib.lx_tune_set(name, M, N, K, v)
                    row.append(f"{lib.lx_tune_variant_name(name, v).decode()} {shape.time('bf16x', M, x_dtype):.2f}")
                lib.lx_tune_clear()
                print(f"|   variants | {'; '.join(row)} |", flush=True)
        del shape
        torch.cuda.empty_cache()


def experts_table():
    E, L, I, top_k, tables = 128, 1024, 512, 4, 8
    gu = [(torch.randn(E, 2 * I, L, device="cuda") * 0.02).to(torch.bfloat16) for _ in range(tables)]
    dn = [(torch.randn(E, L, I, device="cuda") * 0.02).to(torch.bfloat16) for _ in range(tables)]
    gu_x = [encode(w) for w in gu]
    dn_x = [encode(w) for w in dn]
    row_gu, row_dn = gu_x[0].numel() // (E * 2 * I), dn_x[0].numel() // (E * L)
    print()
    print("| T | distinct experts | moe.cu bf16 us | decode bf16 us | decode bf16 GB/s | BF16X us | BF16X GB/s | "
          "BF16X % | vs decode bf16 | vs moe.cu |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for T in (1, 2, 4, 8):
        calls = 4 * tables
        latent = torch.randn(T, L, device="cuda").to(torch.bfloat16)
        ids = [torch.rand(T, E, device="cuda").argsort(-1)[:, :top_k].to(torch.int32).contiguous() for _ in range(calls)]
        weights = torch.softmax(torch.randn(T, top_k, device="cuda"), -1).contiguous()
        out = torch.empty(T, L, device="cuda")
        ws = torch.empty(lib.lx_moe_bf16x_workspace_bytes(T, I, top_k), dtype=torch.uint8, device="cuda")
        distinct = statistics.mean(torch.unique(i).numel() for i in ids)

        def make(kind, i):
            t = i % tables
            if kind == "moe":
                return lambda: ok(lib.lx_moe_experts(latent.data_ptr(), ids[i].data_ptr(), weights.data_ptr(),
                                                     gu[t].data_ptr(), dn[t].data_ptr(), out.data_ptr(), T, L, I,
                                                     top_k, stream()))
            fn = lib.lx_moe_experts_decode_bf16 if kind == "bf16" else lib.lx_moe_experts_bf16x
            g, d = (gu[t], dn[t]) if kind == "bf16" else (gu_x[t], dn_x[t])
            return lambda: ok(fn(latent.data_ptr(), ids[i].data_ptr(), weights.data_ptr(), g.data_ptr(), d.data_ptr(),
                                 out.data_ptr(), ws.data_ptr(), T, L, I, top_k, stream()))

        t_moe = graph_us_per_launch([make("moe", i) for i in range(calls)])
        t_16 = graph_us_per_launch([make("bf16", i) for i in range(calls)])
        t_x = graph_us_per_launch([make("x", i) for i in range(calls)])
        b16 = distinct * 2 * (2 * I * L + L * I)
        bx = distinct * (2 * I * row_gu + L * row_dn)
        print(f"| {T} | {distinct:.1f} | {t_moe:.2f} | {t_16:.2f} | {gbps(b16, t_16):.0f} | {t_x:.2f} | "
              f"{gbps(bx, t_x):.0f} | {100 * gbps(bx, t_x) / DRAM_GBPS:.1f} | {t_16 / t_x:.3f}x | {t_moe / t_x:.3f}x |",
              flush=True)


def main():
    print(f"device: {torch.cuda.get_device_name(0)}; DRAM {DRAM_GBPS:.0f} GB/s; rotation {ROTATE_BYTES >> 20} MB")
    gemv_table("--sweep" in sys.argv)
    experts_table()


if __name__ == "__main__":
    main()
