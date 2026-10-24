"""Times the speculative-decoding kernels against what they replace: lx_spec_gemv_f32 against
lx_gemm_f32 (cuBLASLt) on the decode-step f32 shapes, the verify scan against the plain scan.

    $PY bench/bench_spec.py
"""

import ctypes
import os

import torch

lib = ctypes.CDLL(os.environ["LX_KERNELS"])


def call(name, *args):
    fn = getattr(lib, "lx_" + name)
    conv = []
    for a in args:
        if isinstance(a, torch.Tensor):
            conv.append(ctypes.c_void_p(a.data_ptr()))
        elif isinstance(a, float):
            conv.append(ctypes.c_float(a))
        elif a is None:
            conv.append(ctypes.c_void_p(None))
        else:
            conv.append(ctypes.c_int(a))
    conv.append(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    assert fn(*conv) == 0


def timeit(f, iters=200):
    for _ in range(10):
        f()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        f()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1e3  # microseconds


def i32(v):
    return torch.tensor(v, dtype=torch.int32, device="cuda")


print("f32 products, microseconds per call (M rows)")
for name, N, K in [("router praxis", 128, 1280), ("router logos", 192, 1280), ("depth hidden praxis", 256, 6400),
                   ("depth hidden logos", 256, 7936), ("depth out", 1, 256)]:
    w = torch.randn(N, K, device="cuda")
    b = torch.randn(N, device="cuda")
    for M in (1, 4, 8):
        x = torch.randn(M, K, device="cuda")
        y = torch.empty(M, N, device="cuda")
        t_lt = timeit(lambda: call("gemm_f32", x, K, w, b, y, N, M, N, K))
        t_sp = timeit(lambda: call("spec_gemv_f32", x, K, w, b, y, N, M, N, K))
        print(f"  {name:22s} M={M}  cuBLASLt {t_lt:7.1f}  spec_gemv_f32 {t_sp:7.1f}")

print("Mamba scan, Praxis layer, microseconds per call")
H, P, G, N = 64, 64, 8, 128
hist_floats = lib.lx_spec_ssm_hist_floats
hist_floats.restype = ctypes.c_int64
state = torch.randn(1, H, P, N, device="cuda")
hist = torch.zeros(1, hist_floats(H, P, N), device="cuda")
A_log, D, dt_bias = torch.zeros(H, device="cuda"), torch.zeros(H, device="cuda"), torch.zeros(H, device="cuda")
for T in (1, 4, 7):
    x = torch.randn(T, H * P, device="cuda")
    dt = torch.randn(T, H, device="cuda")
    B = torch.randn(T, G * N, device="cuda")
    C = torch.randn(T, G * N, device="cuda")
    y = torch.empty(T, H, P, device="cuda")
    cu, slot, has = i32([0, T]), i32([0]), i32([1])
    t_plain = timeit(lambda: call("ssm_scan_varlen", x, H * P, dt, H, B, C, G * N, A_log, D, dt_bias, state, slot,
                                  has, cu, 1, y, H, P, G, N))
    for fold, keep in ((0, 0), (0, 1), (8 - T, 0)):
        f, k = i32([fold]), i32([keep])
        t_spec = timeit(lambda: call("spec_ssm_scan", x, H * P, dt, H, B, C, G * N, A_log, D, dt_bias, state, hist,
                                     slot, has, f, k, cu, 1, y, H, P, G, N))
        print(f"  T={T}  plain {t_plain:6.1f}  spec (fold {fold}, keep {keep}) {t_spec:6.1f}")
