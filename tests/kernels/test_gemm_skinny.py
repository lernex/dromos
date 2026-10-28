"""The skinny decode GEMM (engine/cuda/gemm_skinny.cu, lx_gemm_skinny) against PyTorch, and its batch invariance.

The contract is in kernels.h ("Skinny decode GEMM"): each output element is one f32 chain over K in 16-wide k-tiles,
and the bits of an output row depend only on that row of x, the weight and the epilogue. So a row computed alone, with
other rows, or in a launch of any M (chunked above 64) is the same bit pattern. The PyTorch check is a tolerance check
(float64 reference, half a bf16 ulp of the largest output); the invariance check is bitwise.
"""

import ctypes

import pytest
import torch

from conftest import BF16, DTYPE, F32, close
from formats import W_BF16, random_weight

# (N, K): the Logos and Praxis decode products the kernel covers (N a multiple of 32, K of 64).
SHAPES = [
    (10304, 2048),
    (2048, 4096),
    (3072, 2048),
    (2048, 2048),
    (1024, 2048),
    (2048, 1024),
    (65536, 2048),
    (1024, 1024),
    (256, 6400),
    (2560, 2560),
]


def run(lx, x, w, bias, y, alpha, beta, M, N, K):
    bias_dtype = DTYPE[bias.dtype] if bias is not None else F32
    lw = w.lx()
    lx(
        "gemm_skinny", x, DTYPE[x.dtype], K, ctypes.pointer(lw), bias, bias_dtype,
        y, DTYPE[y.dtype], N, M, N, K, float(alpha), float(beta),
    )
    return y


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


def bitwise(a, b):
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


@pytest.mark.parametrize("N,K", SHAPES)
@pytest.mark.parametrize("M", [1, 2, 8, 32, 64, 100])
def test_bf16_matches_reference(lx, N, K, M):
    w = random_weight(W_BF16, N, K, seed=N + K)
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    run(lx, x, w, None, y, 1.0, 0.0, M, N, K)
    check(y, reference(x, w.dense, None, 1.0, 0.0, None))


@pytest.mark.parametrize("N,K", [(2560, 2560), (10304, 2048)])
def test_bias_alpha_beta_f32_output(lx, N, K):
    M = 16
    w = random_weight(W_BF16, N, K, seed=3)
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    bias = torch.randn(N, device="cuda").to(torch.bfloat16)
    y0 = torch.randn(M, N, device="cuda")
    y = y0.clone()
    run(lx, x, w, bias, y, 0.5, 2.0, M, N, K)
    check(y, reference(x, w.dense, bias, 0.5, 2.0, y0))


@pytest.mark.parametrize("N,K", [(2560, 2560), (10304, 2048), (256, 6400)])
def test_rows_are_batch_invariant(lx, N, K):
    """A row's bits do not depend on M, on its position in the launch, or on the other rows."""
    w = random_weight(W_BF16, N, K, seed=7)
    x = torch.randn(100, K, device="cuda").to(torch.bfloat16)
    full = run(lx, x, w, None, torch.empty(100, N, dtype=torch.bfloat16, device="cuda"), 1.0, 0.0, 100, N, K)
    for M in (1, 2, 8, 32, 64):
        y = run(lx, x[:M].contiguous(), w, None, torch.empty(M, N, dtype=torch.bfloat16, device="cuda"), 1.0, 0.0, M, N, K)
        assert bitwise(y, full[:M]), f"M={M}: rows differ from the M=100 launch"
    for r in (0, 1, 37, 63, 64, 99):
        alone = run(lx, x[r : r + 1].contiguous(), w, None, torch.empty(1, N, dtype=torch.bfloat16, device="cuda"),
                    1.0, 0.0, 1, N, K)
        assert bitwise(alone, full[r : r + 1]), f"row {r} alone differs from its row in the M=100 launch"


def test_unsupported_shapes_are_refused(lx):
    """N not a multiple of 32 is outside the kernel's shape contract: the call reports it, it does not guess."""
    w = random_weight(W_BF16, 8, 2048, seed=1)
    x = torch.randn(4, 2048, device="cuda").to(torch.bfloat16)
    y = torch.empty(4, 8, dtype=torch.bfloat16, device="cuda")
    lw = w.lx()
    status = lx.lib.lx_gemm_skinny(
        ctypes.c_void_p(x.data_ptr()), ctypes.c_int(BF16), ctypes.c_int(2048), ctypes.pointer(lw), None, ctypes.c_int(F32),
        ctypes.c_void_p(y.data_ptr()), ctypes.c_int(BF16), ctypes.c_int(8), ctypes.c_int(4), ctypes.c_int(8),
        ctypes.c_int(2048), ctypes.c_float(1.0), ctypes.c_float(0.0), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
    )
    assert status == 801, f"expected LX_ERR_NOT_SUPPORTED (801), got {status}"
