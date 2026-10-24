"""Decode GEMV on every weight format (engine/cuda/gemv.cu, lx_gemv_w) against PyTorch.

    y[m, n] = alpha * (sum_k x[m, k] * w[n, k] + bias[n]) + beta * y[m, n]

The weight is random in its format (formats.py), and the reference is float64 on the exact values the
stored bytes stand for. Tolerances are relative to the largest output magnitude: a bf16 output is
rounded once (2^-8 of its value), an f32 output differs only in summation order (2e-4 is generous).
"""

import ctypes
import json
from pathlib import Path

import pytest
import torch

from conftest import BF16, DTYPE, F32, close
from formats import FORMATS, W_BF16, W_FP8, W_INT8, W_NVFP4, random_weight

FMT_NAMES = list(FORMATS)
METIS_SHAPES = [  # (N, K): the Praxis decode products
    (10304, 2048),
    (2048, 4096),
    (3072, 2048),
    (2048, 2048),
    (1024, 2048),
    (2048, 1024),
    (65536, 2048),
]


def run_w(lx, x, w, bias, y, alpha, beta, M, N, K, ld_x=None, ld_y=None):
    bias_dtype = DTYPE[bias.dtype] if bias is not None else F32
    lw = w.lx()
    lx(
        "gemv_w", x, DTYPE[x.dtype], ld_x if ld_x is not None else K, _ref(lw), bias, bias_dtype,
        y, DTYPE[y.dtype], ld_y if ld_y is not None else N, M, N, K, float(alpha), float(beta),
    )


def _ref(lw):
    """Pass the lx_weight by pointer, as the C signature takes it."""
    import ctypes

    return ctypes.pointer(lw)


def reference(x, dense, bias, alpha, beta, y0):
    out = x.double() @ dense.double().T
    if bias is not None:
        out = out + bias.double()
    out = alpha * out
    if beta != 0.0:
        out = out + beta * y0.double()
    return out


def check(got, want):
    """Within half a bf16 ulp of the largest output (bf16 outputs) or 2e-4 of it (f32 outputs)."""
    scale = want.abs().max().item()
    rel = 2.0**-8 if got.dtype == torch.bfloat16 else 2e-4
    close(got, want, atol=rel * scale + 1e-30)


@pytest.mark.parametrize("K", [48, 528, 2048])  # 48 takes the scalar path, the others the vector path
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (0.5, 0.0), (-1.5, 1.0)])
@pytest.mark.parametrize("y_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fmt", FMT_NAMES)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M", [1, 8])
def test_gemv_w_dtype_matrix(lx, M, x_dtype, fmt, bias_dtype, y_dtype, alpha, beta, K):
    N = 37  # not a multiple of any block's row count
    torch.manual_seed(M * 101 + K + N)
    x = torch.randn(M, K, device="cuda").to(x_dtype)
    w = random_weight(FORMATS[fmt], N, K, seed=M * 7 + K)
    bias = None if bias_dtype is None else torch.randn(N, device="cuda").to(bias_dtype)
    y0 = torch.randn(M, N, device="cuda").to(y_dtype)
    # With beta 0 the kernel must not read y, so the buffer starts as NaN.
    y = y0.clone() if beta != 0.0 else torch.full((M, N), float("nan"), device="cuda", dtype=y_dtype)

    run_w(lx, x, w, bias, y, alpha, beta, M, N, K)
    check(y, reference(x, w.dense, bias, alpha, beta, y0))


@pytest.mark.parametrize("M", list(range(1, 9)))
@pytest.mark.parametrize("fmt", FMT_NAMES)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
def test_gemv_w_every_m(lx, x_dtype, fmt, M):
    N, K = 300, 1024
    torch.manual_seed(M)
    x = torch.randn(M, K, device="cuda").to(x_dtype)
    w = random_weight(FORMATS[fmt], N, K, seed=M + 3)
    bias = torch.randn(N, device="cuda")
    y = torch.empty(M, N, device="cuda")
    run_w(lx, x, w, bias, y, 1.0, 0.0, M, N, K)
    check(y, reference(x, w.dense, bias, 1.0, 0.0, None))


@pytest.mark.parametrize("pad", [0, 3, 8])  # 3 breaks 16-byte alignment of x and y rows, 8 keeps it
@pytest.mark.parametrize("fmt", FMT_NAMES)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M", [1, 5])
def test_gemv_w_strided(lx, M, x_dtype, fmt, pad):
    N, K = 300, 1024
    torch.manual_seed(M * 7 + pad)
    x_buf = torch.randn(M, K + pad, device="cuda").to(x_dtype)
    w = random_weight(FORMATS[fmt], N, K, seed=pad + 9)
    bias = torch.randn(N, device="cuda").to(torch.bfloat16)
    y0 = torch.randn(M, N + pad, device="cuda")
    y_buf = y0.clone()

    run_w(lx, x_buf, w, bias, y_buf, 2.0, 0.5, M, N, K, ld_x=K + pad, ld_y=N + pad)
    want = reference(x_buf[:, :K], w.dense, bias, 2.0, 0.5, y0[:, :N])
    check(y_buf[:, :N], want)
    assert torch.equal(y_buf[:, N:], y0[:, N:]), "wrote past the N columns of y"


@pytest.mark.parametrize("K", [16, 32, 48, 64, 128, 256, 272, 512, 544, 1024])
@pytest.mark.parametrize("fmt", FMT_NAMES)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
def test_gemv_w_k_boundaries(lx, x_dtype, fmt, K):
    # K around the scalar / vector cut and the step sizes (128, 256, 512 k per warp step).
    N, M = 21, 3
    torch.manual_seed(K)
    x = torch.randn(M, K, device="cuda").to(x_dtype)
    w = random_weight(FORMATS[fmt], N, K, seed=K)
    y = torch.empty(M, N, device="cuda")
    run_w(lx, x, w, None, y, 1.0, 0.0, M, N, K)
    check(y, reference(x, w.dense, None, 1.0, 0.0, None))


@pytest.mark.parametrize("N,K", METIS_SHAPES)
@pytest.mark.parametrize("M", [1, 2, 4, 8])
@pytest.mark.parametrize("fmt", FMT_NAMES)
def test_gemv_w_metis_shapes(lx, N, K, M, fmt):
    torch.manual_seed(N * 31 + K + M)
    x = torch.randn(M, K, device="cuda")
    w = random_weight(FORMATS[fmt], N, K, seed=N % 97 + M)
    y = torch.empty(M, N, device="cuda")
    run_w(lx, x, w, None, y, 1.0, 0.0, M, N, K)
    check(y, reference(x, w.dense, None, 1.0, 0.0, None))


@pytest.mark.parametrize("M,N,K", [(1, 256, 2048), (8, 8, 2048), (8, 10304, 2048), (3, 128, 1280), (1, 3072, 8)])
@pytest.mark.parametrize("fmt", FMT_NAMES)
def test_gemv_w_is_deterministic(lx, M, N, K, fmt):
    # No atomics: split-K sums combine in a fixed order, so two runs agree bit for bit.
    if fmt == "nvfp4" and K % 16:
        pytest.skip("NVFP4 rows are whole 16-value blocks")
    torch.manual_seed(M + N)
    x = torch.randn(M, K, device="cuda")
    w = random_weight(FORMATS[fmt], N, K, seed=N)
    first = torch.empty(M, N, device="cuda")
    second = torch.empty(M, N, device="cuda")
    run_w(lx, x, w, None, first, 1.0, 0.0, M, N, K)
    run_w(lx, x, w, None, second, 1.0, 0.0, M, N, K)
    assert torch.equal(first, second)


def test_gemv_w_nvfp4_global_multiplies_once(lx):
    # The global is one factor per row, not per block: doubling it doubles every output exactly.
    N, K, M = 64, 2048, 2
    torch.manual_seed(9)
    x = torch.randn(M, K, device="cuda")
    w = random_weight(W_NVFP4, N, K, seed=4)
    y1 = torch.empty(M, N, device="cuda")
    run_w(lx, x, w, None, y1, 1.0, 0.0, M, N, K)
    w.glob *= 2.0
    y2 = torch.empty(M, N, device="cuda")
    run_w(lx, x, w, None, y2, 1.0, 0.0, M, N, K)
    assert torch.equal(y2, 2.0 * y1)


def test_lx_gemv_wrapper_matches_gemv_w(lx):
    # The original entry point is a thin wrapper: same bytes out for bf16 and int8 weights.
    N, K, M = 300, 1024, 4
    torch.manual_seed(2)
    x = torch.randn(M, K, device="cuda")
    for fmt in (W_BF16, W_INT8):
        w = random_weight(fmt, N, K, seed=fmt + 1)
        a = torch.empty(M, N, device="cuda")
        b = torch.empty(M, N, device="cuda")
        if fmt == W_INT8:
            lx("gemv", x, DTYPE[x.dtype], K, w.data, 1, w.scale, None, F32, a, F32, N, M, N, K, 1.0, 0.0)
        else:
            lx("gemv", x, DTYPE[x.dtype], K, w.data, 0, None, None, F32, a, F32, N, M, N, K, 1.0, 0.0)
        run_w(lx, x, w, None, b, 1.0, 0.0, M, N, K)
        assert torch.equal(a, b)


@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fmt", FMT_NAMES)
def test_gemv_w_every_variant(lx, fmt, x_dtype):
    # The autotuner may pin any built variant for a shape, so each one must compute the same product:
    # row counts that leave partial row groups, a K whose last step is partial (1040 = 4 * 256 + 16,
    # and 1296 = 5 * 256 + 16 in the M = 8 bucket, where both loop styles are built), splits with more
    # parts than steps, and M below its bucket.
    import ctypes

    lib = lx.lib
    setup_tuner(lib)
    kernel = f"gemv_w_{fmt}".encode()
    variants = lib.lx_tune_variants(kernel)
    assert variants > 0
    try:
        for M, N, K in [(1, 300, 1040), (3, 77, 2048), (8, 129, 1024), (5, 50, 1296), (2, 64, 528)]:
            torch.manual_seed(M * 13 + N + K)
            x = torch.randn(M, K, device="cuda").to(x_dtype)
            w = random_weight(FORMATS[fmt], N, K, seed=N + K)
            bias = torch.randn(N, device="cuda")
            want = reference(x, w.dense, bias, 1.0, 0.0, None)
            for v in range(variants):
                lib.lx_tune_clear()
                # Forced: a variant outside the shape's order class is still a kernel to check.
                lib.lx_tune_force(kernel, M, N, K, v)  # ctypes passes the int and bytes as C int and char*
                y = torch.full((M, N), float("nan"), device="cuda")
                run_w(lx, x, w, bias, y, 1.0, 0.0, M, N, K)
                check(y, want)
    finally:
        lib.lx_tune_clear()


TUNING = Path(__file__).resolve().parents[2] / "tuning" / "NVIDIA_GeForce_RTX_5070_Ti_sm120_0.2.json"
LX_ERR_TUNE_REJECTED = 802


def setup_tuner(lib):
    """The tuner's C entry points with their types (lx_tune_set reports a rejected entry)."""
    lib.lx_tune_set.restype = ctypes.c_int
    lib.lx_tune_set.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.lx_tune_clear.restype = None
    lib.lx_tune_clear.argtypes = []
    lib.lx_tune_variants.restype = ctypes.c_int
    lib.lx_tune_variants.argtypes = [ctypes.c_char_p]
    lib.lx_tune_variant_name.restype = ctypes.c_char_p
    lib.lx_tune_variant_name.argtypes = [ctypes.c_char_p, ctypes.c_int]


def use_table(lib, which):
    """Clears the tuner, then for "shipped" applies the 5070 Ti table's GEMV entries the way the engine does."""
    lib.lx_tune_clear()
    if which == "shipped":
        for e in json.loads(TUNING.read_text())["entries"]:
            if e["key"].startswith("gemv_w_"):
                status = lib.lx_tune_set(e["key"].encode(), e["M"], e["N"], e["K"], e["variant"])
                assert status == 0, f"the shipped table rejects {e['key']} M={e['M']} N={e['N']} K={e['K']}"


def variant_split(name):
    """The split S a variant name carries: "s2r4d3g" has S = 2."""
    return int(name[1 : name.index("r")])


@pytest.mark.parametrize("table", ["default", "shipped"])
@pytest.mark.parametrize("N,K", METIS_SHAPES)
@pytest.mark.parametrize("fmt", FMT_NAMES)
def test_rows_do_not_depend_on_the_batch(lx, fmt, N, K, table):
    # The engine's batch invariance: a row's bits depend on (format, N, K), the row and the weight, never
    # on M or on the other rows. Each row computed in a batch of M (1 to 8) equals the same row run alone
    # at M = 1, with the default picks and with the shipped tuning table.
    lib = lx.lib
    setup_tuner(lib)
    use_table(lib, table)
    try:
        w = random_weight(FORMATS[fmt], N, K, seed=N + K, reference=False)
        torch.manual_seed(N + K + 1)
        x = torch.randn(8, K, device="cuda").to(torch.bfloat16)
        alone = torch.empty(8, N, dtype=torch.float32, device="cuda")
        for m in range(8):
            run_w(lx, x[m : m + 1], w, None, alone[m : m + 1], 1.0, 0.0, 1, N, K)
        for M in range(1, 9):
            y = torch.empty(M, N, dtype=torch.float32, device="cuda")
            run_w(lx, x[:M], w, None, y, 1.0, 0.0, M, N, K)
            for m in range(M):
                assert torch.equal(y[m], alone[m]), f"row {m} of an M={M} batch differs from the row alone"
    finally:
        lib.lx_tune_clear()


@pytest.mark.parametrize("N,K", METIS_SHAPES)
@pytest.mark.parametrize("fmt", FMT_NAMES)
def test_order_class_is_the_same_for_every_batch_size(lx, fmt, N, K):
    # The tuner may set a variant at any M, but only the shape's order class is admitted, and the class
    # does not depend on M: the same variants are refused at every batch size, and all admitted ones
    # share one split.
    lib = lx.lib
    setup_tuner(lib)
    key = f"gemv_w_{fmt}".encode()
    count = lib.lx_tune_variants(key)
    admitted = {}
    try:
        for M in (1, 2, 4, 8):
            lib.lx_tune_clear()
            admitted[M] = []
            for v in range(count):
                status = lib.lx_tune_set(key, M, N, K, v)
                assert status in (0, LX_ERR_TUNE_REJECTED)
                if status == 0:
                    admitted[M].append(v)
        assert admitted[1], "no variant is admitted for this shape"
        assert admitted[2] == admitted[4] == admitted[8] == admitted[1]
        splits = {variant_split(lib.lx_tune_variant_name(key, v).decode()) for v in admitted[1]}
        assert len(splits) == 1, f"the admitted variants span splits {sorted(splits)}"
    finally:
        lib.lx_tune_clear()


@pytest.mark.parametrize("M", [1, 2, 4, 8])
def test_bf16x_routes_to_the_lossless_gemv(lx, M):
    # lx_gemv_w on LX_W_BF16X is lx_gemv_bf16x, bit for bit, and that is the product of the bf16 matrix.
    from test_bf16x import encode
    from formats import LxWeight, W_BF16X

    N, K = 1024, 2048
    torch.manual_seed(40 + M)
    w = (torch.randn(N, K, device="cuda") * 0.1).to(torch.bfloat16)
    data = encode(lx, w)
    lw = LxWeight(type=W_BF16X, data=data.data_ptr(), scale=None, global_=1.0, expert_global=None)
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    y = torch.empty(M, N, dtype=torch.float32, device="cuda")
    lx("gemv_w", x, BF16, K, ctypes.pointer(lw), None, F32, y, F32, N, M, N, K, 1.0, 0.0)
    direct = torch.empty(M, N, dtype=torch.float32, device="cuda")
    lx("gemv_bf16x", x, BF16, K, data, None, F32, direct, F32, N, M, N, K, 1.0, 0.0)
    assert torch.equal(y, direct)
    want = x.double() @ w.double().T
    close(y, want.float(), atol=2e-4 * want.abs().max().item())
