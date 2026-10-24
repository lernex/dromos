"""Round-to-nearest quantization (engine/cuda/quant.cu, lx_quantize_weight) against PyTorch.

The rules are kernels.h's: INT8 and FP8 take per-row absmax scales (/127, /448); NVFP4 takes a global
per matrix, max|w| / (448 * 6), and a per-block e4m3 scale, absmax / (6 * global), each value rounded to
the nearest e2m1 of w / (scale * global) with ties to the even code. The references divide tensor by
tensor (IEEE), so the bytes must match exactly, not just within a tolerance.
"""

import ctypes

import pytest
import torch

from conftest import DTYPE, F32, close
from formats import LxWeight, W_BF16, W_FP8, W_INT8, W_NVFP4, _e2m1_values, random_weight, stack_weight

MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def e2m1_codes_ref(v):
    """Nearest e2m1 code, ties to the even code, saturating at 6: the rule of the checkpoints' quantizer."""
    mag = v.abs().clamp(max=6.0)
    idx = torch.bucketize(mag, MIDS.to(v.device))
    tie = (idx < 7) & (mag == MIDS.to(v.device)[idx.clamp(max=6)])
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)
    sign = (v < 0) & (idx > 0)
    return (idx | (sign.to(idx.dtype) << 3)).to(torch.uint8)


def ref_fp8(w):
    wf = w.float()
    amax = wf.abs().amax(-1)
    scale = torch.where(amax > 0, amax / torch.full_like(amax, 448.0), torch.ones_like(amax))
    v = torch.clamp(wf / scale[:, None], -448.0, 448.0)
    return v.to(torch.float8_e4m3fn).view(torch.uint8), scale


def ref_nvfp4(w):
    """(packed codes [N, K/2], block scales as e4m3 bytes [N, K/16], global f32) for bf16 w [N, K]."""
    wf = w.float()
    N, K = wf.shape
    amax_all = wf.abs().max()
    g = amax_all / torch.full_like(amax_all, 2688.0)
    blocks = wf.reshape(N, K // 16, 16)
    bmax = blocks.abs().amax(-1)
    six_g = 6.0 * g
    sb_val = bmax / six_g.expand_as(bmax)
    sb = sb_val.clamp(max=448.0).to(torch.float8_e4m3fn)
    eff = sb.float() * g
    eff3 = eff[..., None].expand_as(blocks)
    nonzero = eff3 > 0
    ratio = torch.where(nonzero, blocks / torch.where(nonzero, eff3, torch.ones_like(eff3)), torch.zeros_like(blocks))
    codes = torch.where(nonzero, e2m1_codes_ref(ratio), torch.zeros_like(ratio, dtype=torch.uint8)).reshape(N, K)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, sb.view(torch.uint8), g


def quantize(lx, w, fmt, out_data, out_scale, glob_out):
    """Calls lx_quantize_weight on the bf16 matrix w into the given buffers; returns the global read back."""
    lw = LxWeight(type=fmt, data=out_data.data_ptr(), scale=out_scale.data_ptr(), global_=1.0, expert_global=None)
    lx("quantize_weight", w, ctypes.pointer(lw), ctypes.c_int64(w.shape[0]), w.shape[1])
    return lw.global_


@pytest.mark.parametrize("rows,cols", [(1, 8), (7, 2048), (300, 1000), (2048, 1024), (131072, 1024)])
def test_int8_matches_quantize_w8(lx, rows, cols):
    torch.manual_seed(rows + cols)
    w = (torch.randn(rows, cols, device="cuda") * torch.rand(rows, 1, device="cuda") * 4).to(torch.bfloat16)
    q = torch.empty(rows, cols, dtype=torch.int8, device="cuda")
    scale = torch.empty(rows, dtype=torch.float32, device="cuda")
    quantize(lx, w, W_INT8, q, scale, None)
    absmax = w.float().abs().amax(-1)
    s_ref = torch.where(absmax > 0, absmax / torch.full_like(absmax, 127.0), torch.ones_like(absmax))
    q_ref = torch.clamp(torch.round(w.float() / s_ref[:, None]), -127, 127).to(torch.int8)
    assert torch.equal(scale, s_ref)
    assert torch.equal(q, q_ref)


@pytest.mark.parametrize("rows,cols", [(1, 16), (7, 2048), (300, 1000), (2048, 1024), (131072, 1024)])
def test_fp8_matches_reference(lx, rows, cols):
    torch.manual_seed(rows * 3 + cols)
    w = (torch.randn(rows, cols, device="cuda") * torch.rand(rows, 1, device="cuda") * 40).to(torch.bfloat16)
    if rows > 2:
        w[0] = 0.0  # an all-zero row keeps scale 1
    q = torch.empty(rows, cols, dtype=torch.uint8, device="cuda")
    scale = torch.empty(rows, dtype=torch.float32, device="cuda")
    quantize(lx, w, W_FP8, q, scale, None)
    q_ref, s_ref = ref_fp8(w)
    assert torch.equal(scale, s_ref), "row scales differ from absmax / 448"
    assert torch.equal(q, q_ref), "e4m3 bytes differ from the reference"


@pytest.mark.parametrize("rows,cols", [(1, 16), (7, 256), (300, 1024), (2048, 2048), (1024, 2048), (4096, 512)])
def test_nvfp4_matches_reference(lx, rows, cols):
    torch.manual_seed(rows + cols * 7)
    w = (torch.randn(rows, cols, device="cuda") * torch.rand(rows, cols // 16, device="cuda").repeat_interleave(16, 1) * 3).to(torch.bfloat16)
    w[:, :16] = 0.0  # a block of zeros: block scale 0, codes 0
    data = torch.empty(rows, cols // 2, dtype=torch.uint8, device="cuda")
    scale = torch.empty(rows, cols // 16, dtype=torch.uint8, device="cuda")
    g = quantize(lx, w, W_NVFP4, data, scale, None)
    packed_ref, scale_ref, g_ref = ref_nvfp4(w)
    assert g == float(g_ref), "global differs from max|w| / (448 * 6)"
    assert torch.equal(scale, scale_ref), "block scales differ"
    assert torch.equal(data, packed_ref), "e2m1 codes differ from the reference"


def test_nvfp4_ties_and_signs(lx):
    # The block with amax 2688 sets the global to 1. Blocks with amax 12 get scale 2, and their values
    # w / 2 land on midpoints of the e2m1 grid: ties go to the even code (0, 1, 1, 2, 2, 4, 4).
    cols = 16
    w = torch.zeros(2, cols, device="cuda")
    w[0, 0] = 2688.0
    w[1, 0] = 12.0
    w[1, 1:8] = torch.tensor([0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 10.0], device="cuda")
    w[1, 8:15] = -torch.tensor([0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 10.0], device="cuda")
    w = w.to(torch.bfloat16)
    data = torch.empty(2, 8, dtype=torch.uint8, device="cuda")
    scale = torch.empty(2, 1, dtype=torch.uint8, device="cuda")
    g = quantize(lx, w, W_NVFP4, data, scale, None)
    assert g == 1.0
    codes = torch.stack([data & 15, data >> 4], dim=-1).flatten(-2)
    assert codes[1, 1:8].tolist() == [0, 2, 2, 4, 4, 6, 6], "positive ties"
    # -0.25 rounds to zero: code 0 with no sign, as stored.
    assert codes[1, 8:15].tolist() == [0, 8 + 2, 8 + 2, 8 + 4, 8 + 4, 8 + 6, 8 + 6], "negative ties"
    # Its e4m3 block scale is exactly 2 (0x40).
    assert scale[1, 0].item() == 0x40


def test_nvfp4_stack_per_expert_globals(lx):
    # An [E, N, K] stack quantized one expert at a time: each expert has its own global.
    E, N, K = 4, 64, 256
    torch.manual_seed(3)
    stack = torch.randn(E, N, K, device="cuda").to(torch.bfloat16) * torch.tensor([0.1, 1.0, 3.0, 0.3], device="cuda")[:, None, None].to(torch.bfloat16)
    data = torch.empty(E, N, K // 2, dtype=torch.uint8, device="cuda")
    scale = torch.empty(E, N, K // 16, dtype=torch.uint8, device="cuda")
    globals_ = []
    for e in range(E):
        globals_.append(quantize(lx, stack[e], W_NVFP4, data[e], scale[e], None))
    for e in range(E):
        packed_ref, scale_ref, g_ref = ref_nvfp4(stack[e])
        assert globals_[e] == float(g_ref)
        assert torch.equal(data[e], packed_ref)
        assert torch.equal(scale[e], scale_ref)
    assert len(set(globals_)) == E, "the experts should have distinct globals in this test"


@pytest.mark.parametrize("fmt", [W_FP8, W_NVFP4])
def test_quantized_weight_gemv_error_is_small(lx, fmt):
    # Quantize a bf16 matrix, multiply by the kernel, and compare with the exact bf16 product: the error is
    # the quantization error, which must be small relative to the output (per-element error is ~5% for
    # e2m1 with block scales, ~2% for e4m3).
    N, K, M = 512, 2048, 4
    torch.manual_seed(11)
    w = (torch.randn(N, K, device="cuda") * 0.05).to(torch.bfloat16)
    x = torch.randn(M, K, device="cuda")
    if fmt == W_FP8:
        data = torch.empty(N, K, dtype=torch.uint8, device="cuda")
        scale = torch.empty(N, dtype=torch.float32, device="cuda")
    else:
        data = torch.empty(N, K // 2, dtype=torch.uint8, device="cuda")
        scale = torch.empty(N, K // 16, dtype=torch.uint8, device="cuda")
    glob = quantize(lx, w, fmt, data, scale, None)
    lw = LxWeight(type=fmt, data=data.data_ptr(), scale=scale.data_ptr(), global_=glob, expert_global=None)
    y = torch.empty(M, N, dtype=torch.float32, device="cuda")
    lx("gemv_w", x, F32, K, ctypes.pointer(lw), None, F32, y, F32, N, M, N, K, 1.0, 0.0)
    exact = x.double() @ w.double().T
    rel = ((y.double() - exact).norm() / exact.norm()).item()
    assert rel < (0.12 if fmt == W_NVFP4 else 0.03), f"relative error {rel:.4f}"


def test_random_roundtrip_matches_the_dense_reference(lx):
    # The bytes that quantize writes are the ones the dense reference of formats.py decodes, so a
    # weight built by quantize multiplies exactly like its own dequantized values.
    N, K = 96, 512
    torch.manual_seed(5)
    w = (torch.randn(N, K, device="cuda") * 0.2).to(torch.bfloat16)
    data = torch.empty(N, K // 2, dtype=torch.uint8, device="cuda")
    scale = torch.empty(N, K // 16, dtype=torch.uint8, device="cuda")
    glob = quantize(lx, w, W_NVFP4, data, scale, None)
    from formats import _nvfp4_dense

    dense = _nvfp4_dense(data, scale, glob)
    x = torch.randn(3, K, device="cuda")
    lw = LxWeight(type=W_NVFP4, data=data.data_ptr(), scale=scale.data_ptr(), global_=glob, expert_global=None)
    y = torch.empty(3, N, dtype=torch.float32, device="cuda")
    lx("gemv_w", x, F32, K, ctypes.pointer(lw), None, F32, y, F32, N, 3, N, K, 1.0, 0.0)
    want = x.double() @ dense.T
    close(y, want.float(), atol=2e-4 * want.abs().max().item())


@pytest.mark.parametrize("fmt", [W_BF16, W_INT8, W_FP8, W_NVFP4])
@pytest.mark.parametrize("N,K", [(1, 16), (33, 512), (96, 1024)])
def test_dequantize_matches_the_dense_reference(lx, fmt, N, K):
    # lx_dequantize_weight writes the values each format encodes, rounded once to bf16: every element
    # is within half a bf16 ulp of the float64 reference that formats.py builds from the same bytes.
    w = random_weight(fmt, N, K, seed=11 + fmt + N)
    out = torch.empty(N, K, dtype=torch.bfloat16, device="cuda")
    lx("dequantize_weight", ctypes.pointer(w.lx()), out, ctypes.c_int64(N), K)
    close(out.float(), w.dense.float(), atol=1e-30, rtol=2.0**-8)


def test_dequantize_nvfp4_global_multiplies_once(lx):
    # The global is applied once, in the dequantized values: halving it halves every output exactly.
    N, K = 64, 256
    w = random_weight(W_NVFP4, N, K, seed=3)
    a = torch.empty(N, K, dtype=torch.bfloat16, device="cuda")
    lx("dequantize_weight", ctypes.pointer(w.lx()), a, ctypes.c_int64(N), K)
    half = LxWeight(type=W_NVFP4, data=w.data.data_ptr(), scale=w.scale.data_ptr(), global_=w.glob * 0.5,
                    expert_global=None)
    b = torch.empty(N, K, dtype=torch.bfloat16, device="cuda")
    lx("dequantize_weight", ctypes.pointer(half), b, ctypes.c_int64(N), K)
    assert torch.equal(b.float(), a.float() * 0.5)
