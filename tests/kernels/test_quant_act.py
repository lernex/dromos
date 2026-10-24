"""Per-token activation quantization (engine/cuda/quant_act.cu) against the PyTorch reference in
wformats.py, bit for bit: the quantized values, the per-row scales and the FP4 block scales."""

import pytest
import torch

from conftest import DEVICE, DTYPE
from wformats import MATH_FP4, MATH_FP8, MATH_INT8, quantize_act

SHAPES = [(1, 2048), (7, 1024), (300, 512), (64, 6400), (5, 4096), (3, 16)]


def run(lx, x, ld_x, math, M, K, ld_q, ld_bs):
    q = torch.full((M, ld_q), 0xA5, dtype=torch.uint8, device=DEVICE)
    scale = torch.full((M,), float("nan"), device=DEVICE)
    bs = torch.full((M, max(ld_bs, 1)), 0xA5, dtype=torch.uint8, device=DEVICE)
    lx("quantize_act", x, DTYPE[x.dtype], ld_x, math, q, ld_q, scale, bs if math == MATH_FP4 else None, ld_bs, M, K)
    return q, scale, bs


@pytest.mark.parametrize("pad", [0, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("math", [MATH_INT8, MATH_FP8, MATH_FP4])
@pytest.mark.parametrize("M,K", SHAPES)
def test_quantize_act(lx, M, K, math, dtype, pad):
    torch.manual_seed(M * 31 + K + math)
    x_buf = (torch.randn(M, K + pad, device=DEVICE) * torch.rand(M, 1, device=DEVICE) * 4).to(dtype)
    x_buf[0, :K] *= 1e-3  # a quiet row: small scales, subnormal e4m3 block scales for FP4
    if M > 2:
        x_buf[2, :K] = 0  # a zero row
    x = x_buf[:, :K]
    ld_q = K // 2 + 16 if math == MATH_FP4 else K + 16
    ld_bs = K // 16 + 4 if math == MATH_FP4 else 0
    q, scale, bs = run(lx, x_buf, K + pad, math, M, K, ld_q, ld_bs)
    rq, rscale, rbs = quantize_act(x, math)
    width = rq.shape[1]
    assert torch.equal(q[:, :width], rq), f"quantized values differ at {(q[:, :width] != rq).nonzero()[:4].tolist()}"
    assert torch.equal(scale, rscale.float()), "per-row scales differ"
    assert (q[:, width:] == 0xA5).all(), "wrote past the row"
    if math == MATH_FP4:
        assert torch.equal(bs[:, : K // 16], rbs), "block scales differ"
        assert (bs[:, K // 16 :] == 0xA5).all()
    if M > 2:
        assert (q[2, :width] == 0).all() and scale[2].item() == 1.0, "a zero row quantizes to 0 with scale 1"


@pytest.mark.parametrize("math", [MATH_INT8, MATH_FP8])
def test_quantize_act_odd_k(lx, math):
    # K not a multiple of 16, and a misaligned row stride: the scalar tail path.
    torch.manual_seed(5)
    M, K = 9, 1001
    x_buf = torch.randn(M, K + 3, device=DEVICE)
    q, scale, _ = run(lx, x_buf, K + 3, math, M, K, K + 7, 0)
    rq, rscale, _ = quantize_act(x_buf[:, :K], math)
    assert torch.equal(q[:, :K], rq)
    assert torch.equal(scale, rscale.float())


def test_quantize_act_fp4_extremes(lx):
    # Ties of the e2m1 rounding, saturation, and blocks whose scale underflows to an e4m3 subnormal.
    M, K = 4, 64
    x = torch.zeros(M, K, device=DEVICE)
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, -0.25, -0.75, -2.5, -5.0, 0.1, 7.0, -7.0, 3.0])
    x[0, :16] = ties.to(DEVICE)  # block absmax 7 relative to the row: values land on the e2m1 grid's midpoints
    x[1, :16] = 1.0
    x[1, 16:32] = 1e-6  # a block 10^6 below the row max: its scale underflows
    x[2] = torch.linspace(-3, 3, K, device=DEVICE)
    q, scale, bs = run(lx, x, K, MATH_FP4, M, K, K // 2, K // 16)
    rq, rscale, rbs = quantize_act(x, MATH_FP4)
    assert torch.equal(q, rq)
    assert torch.equal(scale, rscale.float())
    assert torch.equal(bs[:, : K // 16], rbs)
