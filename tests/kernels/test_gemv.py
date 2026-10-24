"""Decode GEMV (engine/cuda/gemv.cu, lx_gemv) and the int8 quantizer (lx_quantize_w8) against PyTorch.

    y[m, n] = alpha * (sum_k x[m, k] * w[n, k] + bias[n]) + beta * y[m, n]

Weights are bf16, or int8 with a per-row f32 scale: w[n, k] = q[n, k] * scale[n]. References run in
float64. Tolerances are relative to the largest output magnitude. A bf16 output is rounded once, so it
is within half a bf16 ulp, 2^-8 of its own value, of the exact result. An f32 output differs from the
reference only in summation order, so 2e-4 is generous.
"""

import ctypes

import pytest
import torch

from conftest import DEVICE, DTYPE, F32, close

W_BF16, W_INT8 = 0, 1
METIS_SHAPES = [  # (N, K)
    (10304, 2048),
    (3072, 2048),
    (2048, 4096),
    (65536, 2048),
    (1024, 2048),
    (2048, 1024),
    (256, 6400),
    (256, 2048),
    (256, 256),
    (24, 64),
    (8, 2048),
    (3072, 8),
    (128, 1280),
    (1, 256),
]


def make_weights(N, K, w_type):
    if w_type == "bf16":
        return torch.randn(N, K, device=DEVICE).to(torch.bfloat16), None
    q = torch.randint(-127, 128, (N, K), device=DEVICE).to(torch.int8)
    scale = (torch.rand(N, device=DEVICE) * 0.02 + 0.001).float()
    return q, scale


def run_gemv(lx, x, w, w_scale, bias, y, alpha, beta, M, N, K, ld_x, ld_y):
    w_type = W_INT8 if w.dtype == torch.int8 else W_BF16
    bias_dtype = DTYPE[bias.dtype] if bias is not None else F32
    lx("gemv", x, DTYPE[x.dtype], ld_x, w, w_type, w_scale, bias, bias_dtype, y, DTYPE[y.dtype], ld_y, M, N, K,
       float(alpha), float(beta))


def reference(x, w, w_scale, bias, alpha, beta, y0):
    """float64, the weight dequantized as q * scale, computed in row blocks to bound the memory."""
    N = w.shape[0]
    out = torch.empty(x.shape[0], N, dtype=torch.float64, device=x.device)
    xd = x.double()
    for n0 in range(0, N, 8192):
        wb = w[n0 : n0 + 8192].double()
        if w_scale is not None:
            wb = wb * w_scale[n0 : n0 + 8192].double()[:, None]
        out[:, n0 : n0 + 8192] = xd @ wb.T
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


@pytest.mark.parametrize("K", [40, 528])  # 40 takes the scalar path, 528 the vector path for both weight types
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (0.5, 0.0), (-1.5, 1.0)])
@pytest.mark.parametrize("y_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("w_type", ["bf16", "int8"])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M", [1, 8])
def test_gemv_dtype_matrix(lx, M, x_dtype, w_type, bias_dtype, y_dtype, alpha, beta, K):
    N = 37  # not a multiple of any block's row count
    torch.manual_seed(M * 101 + K + N)
    x = torch.randn(M, K, device=DEVICE).to(x_dtype)
    w, w_scale = make_weights(N, K, w_type)
    bias = None if bias_dtype is None else torch.randn(N, device=DEVICE).to(bias_dtype)
    y0 = torch.randn(M, N, device=DEVICE).to(y_dtype)
    # With beta 0 the kernel must not read y, so the buffer starts as NaN.
    y = y0.clone() if beta != 0.0 else torch.full((M, N), float("nan"), device=DEVICE, dtype=y_dtype)

    run_gemv(lx, x, w, w_scale, bias, y, alpha, beta, M, N, K, ld_x=K, ld_y=N)
    check(y, reference(x, w, w_scale, bias, alpha, beta, y0))


@pytest.mark.parametrize("M", list(range(1, 9)))
@pytest.mark.parametrize("w_type", ["bf16", "int8"])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
def test_gemv_every_m(lx, x_dtype, w_type, M):
    N, K = 300, 1024
    torch.manual_seed(M)
    x = torch.randn(M, K, device=DEVICE).to(x_dtype)
    w, w_scale = make_weights(N, K, w_type)
    bias = torch.randn(N, device=DEVICE)
    y = torch.empty(M, N, device=DEVICE)
    run_gemv(lx, x, w, w_scale, bias, y, 1.0, 0.0, M, N, K, ld_x=K, ld_y=N)
    check(y, reference(x, w, w_scale, bias, 1.0, 0.0, None))


@pytest.mark.parametrize("pad", [0, 3, 8])  # 3 breaks 16-byte alignment of x and y rows, 8 keeps it
@pytest.mark.parametrize("w_type", ["bf16", "int8"])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M", [1, 5])
def test_gemv_strided(lx, M, x_dtype, w_type, pad):
    N, K = 300, 1024
    torch.manual_seed(M * 7 + pad)
    x_buf = torch.randn(M, K + pad, device=DEVICE).to(x_dtype)
    w, w_scale = make_weights(N, K, w_type)
    bias = torch.randn(N, device=DEVICE).to(torch.bfloat16)
    y0 = torch.randn(M, N + pad, device=DEVICE)
    y_buf = y0.clone()

    run_gemv(lx, x_buf, w, w_scale, bias, y_buf, 2.0, 0.5, M, N, K, ld_x=K + pad, ld_y=N + pad)
    want = reference(x_buf[:, :K], w, w_scale, bias, 2.0, 0.5, y0[:, :N])
    check(y_buf[:, :N], want)
    assert torch.equal(y_buf[:, N:], y0[:, N:]), "wrote past the N columns of y"


@pytest.mark.parametrize("K", [1, 5, 8, 13, 24, 64, 100, 128, 256, 257, 300])
@pytest.mark.parametrize("w_type", ["bf16", "int8"])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
def test_gemv_odd_and_small_k(lx, x_dtype, w_type, K):
    N, M = 21, 3
    torch.manual_seed(K)
    x = torch.randn(M, K, device=DEVICE).to(x_dtype)
    w, w_scale = make_weights(N, K, w_type)
    y = torch.empty(M, N, device=DEVICE)
    run_gemv(lx, x, w, w_scale, None, y, 1.0, 0.0, M, N, K, ld_x=K, ld_y=N)
    check(y, reference(x, w, w_scale, None, 1.0, 0.0, None))


@pytest.mark.parametrize("N,K", METIS_SHAPES)
@pytest.mark.parametrize("M", [1, 8])
@pytest.mark.parametrize("w_type", ["bf16", "int8"])
def test_gemv_metis_shapes(lx, N, K, M, w_type):
    torch.manual_seed(N * 31 + K + M)
    x = torch.randn(M, K, device=DEVICE)
    w, w_scale = make_weights(N, K, w_type)
    y = torch.empty(M, N, device=DEVICE)
    run_gemv(lx, x, w, w_scale, None, y, 1.0, 0.0, M, N, K, ld_x=K, ld_y=N)
    check(y, reference(x, w, w_scale, None, 1.0, 0.0, None))


@pytest.mark.parametrize("M,N,K", [(1, 256, 2048), (8, 8, 2048), (8, 10304, 2048), (3, 128, 1280), (1, 3072, 8)])
@pytest.mark.parametrize("w_type", ["bf16", "int8"])
def test_gemv_is_deterministic(lx, M, N, K, w_type):
    # No atomics: split-K sums combine in a fixed order, so two runs agree bit for bit.
    torch.manual_seed(M + N)
    x = torch.randn(M, K, device=DEVICE)
    w, w_scale = make_weights(N, K, w_type)
    first = torch.empty(M, N, device=DEVICE)
    second = torch.empty(M, N, device=DEVICE)
    run_gemv(lx, x, w, w_scale, None, first, 1.0, 0.0, M, N, K, ld_x=K, ld_y=N)
    run_gemv(lx, x, w, w_scale, None, second, 1.0, 0.0, M, N, K, ld_x=K, ld_y=N)
    assert torch.equal(first, second)


def test_gemv_int8_is_exactly_the_dequantized_product(lx):
    # Scales are applied once per row: a row of q with scale s must match x @ (q * s) in float64.
    N, K, M = 64, 2048, 2
    torch.manual_seed(5)
    x = torch.randn(M, K, device=DEVICE)
    q = torch.full((N, K), 127, dtype=torch.int8, device=DEVICE)
    q[::2] = -127
    scale = torch.linspace(0.5, 2.0, N, device=DEVICE)
    y = torch.empty(M, N, device=DEVICE)
    run_gemv(lx, x, q, scale, None, y, 1.0, 0.0, M, N, K, ld_x=K, ld_y=N)
    check(y, reference(x, q, scale, None, 1.0, 0.0, None))


def quant_reference(w):
    """Per-row symmetric int8 as the spec states it. Divisions are tensor by tensor, so IEEE: torch
    rewrites a division by a Python scalar as a multiply by its reciprocal, which can differ in the last bit."""
    wf = w.float()
    absmax = wf.abs().amax(-1)
    scale = torch.where(absmax > 0, absmax / torch.full_like(absmax, 127.0), torch.ones_like(absmax))
    q = torch.clamp(torch.round(wf / scale[:, None]), -127, 127).to(torch.int8)
    return q, scale


@pytest.mark.parametrize("rows,cols", [(1, 8), (7, 2048), (300, 1000), (2048, 1024), (131072, 1024)])
def test_quantize_w8_matches_reference(lx, rows, cols):
    torch.manual_seed(rows + cols)
    w = (torch.randn(rows, cols, device=DEVICE) * torch.rand(rows, 1, device=DEVICE) * 4).to(torch.bfloat16)
    q = torch.empty(rows, cols, dtype=torch.int8, device=DEVICE)
    scale = torch.empty(rows, dtype=torch.float32, device=DEVICE)
    lx("quantize_w8", w, q, scale, ctypes.c_int64(rows), cols)
    q_ref, s_ref = quant_reference(w)
    assert torch.equal(scale, s_ref), "row scales differ from absmax / 127"
    assert torch.equal(q, q_ref), "quantized values differ from the reference"


def test_quantize_w8_ties_round_half_even_and_zero_rows(lx):
    w = torch.tensor(
        [
            [127.0, 0.5, 1.5, 2.5, -2.5, 3.5, -126.5, 0.0],  # scale 1: halves round to even
            [0.0] * 8,  # all zero: scale 1, q 0
            [254.0, 1.0, 3.0, -5.0, 0.0, 0.0, 0.0, 0.0],  # scale 2: 0.5 and 1.5 and -2.5 are ties
        ],
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    q = torch.empty(3, 8, dtype=torch.int8, device=DEVICE)
    scale = torch.empty(3, dtype=torch.float32, device=DEVICE)
    lx("quantize_w8", w, q, scale, ctypes.c_int64(3), 8)
    assert scale.tolist() == [1.0, 1.0, 2.0]
    assert q.tolist() == [
        [127, 0, 2, 2, -2, 4, -126, 0],
        [0] * 8,
        [127, 0, 2, -2, 0, 0, 0, 0],
    ]
