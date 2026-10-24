"""Dense GEMMs against PyTorch: y = alpha * (x @ w^T + bias) + beta * y, with w stored [N, K].

Every case writes into a y buffer wider than N (and reads x from one wider than K) when `pad` is
set, so a wrong leading dimension shows up as a mismatch, and the padding columns must come back
untouched. pad 3 gives odd row strides, which the cuBLASLt alignment preference has to handle.
"""

import pytest
import torch

from conftest import DEVICE, DTYPE, close

SHAPES = [
    (1, 10304, 2048),  # decode: the Mamba in_proj
    (7, 2048, 4096),
    (300, 3072, 2048),  # prefill
    (5, 128, 1280),
    (1, 256, 6400),
    (33, 24, 64),
]
SCALARS = [(1.0, 0.0), (0.5, 0.0), (1.0, 1.0), (0.5, 1.0)]  # (alpha, beta)
PADS = [0, 8, 3]


def reference(x, w, bias, alpha, beta, y0):
    """alpha * (x @ w^T + bias) + beta * y0 in float32, the way the spec states it."""
    out = x.float() @ w.float().T
    if bias is not None:
        out = out + bias.float()
    out = alpha * out
    if beta != 0.0:
        out = out + beta * y0.float()
    return out


@pytest.mark.parametrize("pad", PADS)
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("alpha,beta", SCALARS)
@pytest.mark.parametrize("y_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_gemm_bf16(lx, M, N, K, y_dtype, alpha, beta, with_bias, pad):
    torch.manual_seed(M * 7919 + N * 31 + K)
    x_buf = torch.randn(M, K + pad, device=DEVICE).to(torch.bfloat16)
    w = torch.randn(N, K, device=DEVICE).to(torch.bfloat16)
    bias = torch.randn(N, device=DEVICE).to(torch.bfloat16) if with_bias else None
    y_prefill = torch.randn(M, N + pad, device=DEVICE).to(y_dtype)
    y_buf = y_prefill.clone()

    lx("gemm_bf16", x_buf, K + pad, w, bias, y_buf, DTYPE[y_dtype], N + pad, M, N, K, alpha, beta)

    x = x_buf[:, :K]
    want = reference(x, w, bias, alpha, beta, y_prefill[:, :N])
    got = y_buf[:, :N]
    if y_dtype == torch.bfloat16:
        # One rounding to bf16 on the way out: half an ulp is 2^-9, so 1e-2 of the result's scale is loose enough.
        close(got, want, atol=1e-2 * want.abs().max().item())
    else:
        close(got, want, atol=2e-3 * want.abs().max().item())
    if pad:
        assert torch.equal(y_buf[:, N:].float(), y_prefill[:, N:].float()), "wrote past the N columns of y"


@pytest.mark.parametrize("pad", PADS)
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_gemm_f32(lx, M, N, K, with_bias, pad):
    torch.manual_seed(M * 7919 + N * 31 + K + 1)
    x_buf = torch.randn(M, K + pad, device=DEVICE)
    w = torch.randn(N, K, device=DEVICE)
    bias = torch.randn(N, device=DEVICE) if with_bias else None
    y_buf = torch.full((M, N + pad), float("nan"), device=DEVICE)

    lx("gemm_f32", x_buf, K + pad, w, bias, y_buf, N + pad, M, N, K)

    want = reference(x_buf[:, :K], w, bias, 1.0, 0.0, None)
    # Full FP32: only the summation order differs from PyTorch's cuBLAS, so 1e-5 of the scale is tight.
    close(y_buf[:, :N], want, atol=1e-5 * want.abs().max().item())
    if pad:
        assert torch.isnan(y_buf[:, N:]).all(), "wrote past the N columns of y"
