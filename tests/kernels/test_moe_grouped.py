"""lx_moe_experts_grouped (engine/cuda/moe_grouped.cu) against PyTorch, for every weight format x math
pair and every kernel family (the card's own, and the sm_89 and sm_80 families through
lx_set_arch_override), at Praxis's and Logos's expert shapes and at small and odd ones.

The reference follows the kernel's arithmetic in float64: the latent as the math sees it (bf16, or
quantized per token with the reference quantizer that test_quant_act checks bit for bit), the experts'
weights dequantized from their stored format, the SwiGLU intermediate rounded to bf16, then that
intermediate as the math sees it (quantized per pair row) through the down projection, weighted and summed
per token. What is left is float32 accumulation order, the kernel's fast exp in silu, and the bf16 and
quantization roundings those can tip, so the bound is relative to the output's magnitude: 2e-3, and 6e-3
under FP4 math (an e2m1 step is a quarter of its block's scale, so a tipped intermediate moves further)."""

import ctypes

import pytest
import torch
import torch.nn.functional as F

from conftest import BACKEND, DEVICE, close
from wformats import (
    MATH_BF16,
    MATH_FP4,
    MATH_FP8,
    MATH_INT8,
    W_BF16,
    W_FP8,
    W_INT8,
    W_NVFP4,
    Weight,
    act_operand,
    dequant_weight,
    quantize_weight,
)

NOT_SUPPORTED = 801

# (name, weight format, math, oldest family that runs it)
PAIRS = [
    ("bf16", W_BF16, MATH_BF16, 80),
    ("w8a16_int8", W_INT8, MATH_BF16, 80),
    ("w8a16_fp8", W_FP8, MATH_BF16, 80),
    ("w4a16", W_NVFP4, MATH_BF16, 80),
    ("int8", W_INT8, MATH_INT8, 80),
    ("fp8", W_FP8, MATH_FP8, 89),
    ("w4a8", W_NVFP4, MATH_FP8, 120),
    ("fp4", W_NVFP4, MATH_FP4, 120),
]
ARCHS = [0, 89, 80]
# (T, L, I, E, top_k)
SHAPES = [
    (5, 256, 256, 8, 2),
    (37, 512, 256, 16, 3),  # partial tiles everywhere
    (200, 1024, 512, 128, 4),  # Praxis
    (96, 1024, 768, 192, 4),  # Logos
]


def set_arch(lx, arch):
    lx.lib.lx_set_arch_override.restype = ctypes.c_int
    return lx.lib.lx_set_arch_override(arch)


def native_arch(lx):
    set_arch(lx, 0)
    return lx.lib.lx_arch()


def workspace_bytes(lx, T, L, I, E, top_k, math):
    fn = lx.lib.lx_moe_grouped_workspace_bytes
    fn.restype = ctypes.c_int64
    return fn(T, L, I, E, top_k, math)


def call_grouped(lx, latent, ids, weights, gu, dn, math, out, ws, T, L, I, E, top_k):
    fn = lx.lib.lx_moe_experts_grouped
    fn.restype = ctypes.c_int
    p = lambda t: ctypes.c_void_p(t.data_ptr())
    status = fn(p(latent), p(ids), p(weights), ctypes.byref(gu.struct), ctypes.byref(dn.struct), math, p(out), p(ws),
                T, L, I, E, top_k, lx.stream)
    lx.sync()
    return status


def expert(w, e):
    """Expert e of a stacked weight, dequantized to float64 (one expert at a time keeps memory small)."""
    scale = w.scale[e] if w.scale is not None else None
    return dequant_weight(Weight(w.wtype, w.data[e], scale, w.glob))


def reference(latent, ids, weights, gu, dn, math):
    """Float64, the kernel's operands: see the module docstring."""
    T, L = latent.shape
    top_k = ids.shape[1]
    x = act_operand(latent, math)
    I = gu.data.shape[1] // 2
    flat = ids.reshape(-1).long()
    mid = torch.empty(T * top_k, I, dtype=torch.bfloat16, device=latent.device)
    for e in flat.unique().tolist():
        rows = (flat == e).nonzero().squeeze(1)
        xe = x[rows // top_k]
        w = expert(gu, e)
        mid[rows] = (F.silu(xe @ w[:I].T) * (xe @ w[I:].T)).to(torch.bfloat16)
    h = act_operand(mid, math)
    y = torch.empty(T * top_k, L, dtype=torch.float64, device=latent.device)
    for e in flat.unique().tolist():
        rows = (flat == e).nonzero().squeeze(1)
        y[rows] = h[rows] @ expert(dn, e).T
    y = y * weights.reshape(-1, 1).double()
    return y.view(T, top_k, L).sum(1)


def make_case(T, L, I, E, top_k, wtype, seed, skew=False):
    torch.manual_seed(seed)
    latent = torch.randn(T, L, device=DEVICE).to(torch.bfloat16)
    if skew:
        # Most tokens on the same few experts: experts with several tiles of rows next to empty ones.
        hot = torch.randperm(E, device=DEVICE)[: max(top_k, 3)]
        noise = torch.rand(T, E, device=DEVICE)
        noise[:, hot] += 10.0 * (torch.rand(T, 1, device=DEVICE) < 0.9)
        ids = noise.argsort(-1, descending=True)[:, :top_k]
    else:
        ids = torch.rand(T, E, device=DEVICE).argsort(-1)[:, :top_k]
    ids = ids.to(torch.int32).contiguous()
    weights = torch.softmax(torch.randn(T, top_k, device=DEVICE), -1).contiguous()
    gu = quantize_weight(torch.randn(E, 2 * I, L, device=DEVICE) / L**0.5, wtype)
    dn = quantize_weight(torch.randn(E, L, I, device=DEVICE) / I**0.5, wtype)
    return latent, ids, weights, gu, dn


def run(lx, case, math, T, L, I, E, top_k):
    latent, ids, weights, gu, dn = case
    out = torch.full((T, L), float("nan"), device=DEVICE)
    ws = torch.full((max(workspace_bytes(lx, T, L, I, E, top_k, math), 1),), 0xFF, dtype=torch.uint8, device=DEVICE)
    return call_grouped(lx, latent, ids, weights, gu, dn, math, out, ws, T, L, I, E, top_k), out


def check(out, case, math):
    latent, ids, weights, gu, dn = case
    want = reference(latent, ids, weights, gu, dn, math)
    tol = (6e-3 if math == MATH_FP4 else 2e-3) * want.abs().max().item()
    close(out, want.float(), atol=tol)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("T,L,I,E,top_k", SHAPES)
@pytest.mark.parametrize("name,wtype,math,oldest", PAIRS)
def test_moe_grouped(lx, name, wtype, math, oldest, T, L, I, E, top_k, arch):
    family = arch if arch else native_arch(lx)
    case = make_case(T, L, I, E, top_k, wtype, seed=T * 31 + E + wtype * 7 + math * 101)
    try:
        set_arch(lx, arch)
        status, out = run(lx, case, math, T, L, I, E, top_k)
    finally:
        set_arch(lx, 0)
    if family < oldest:
        assert status == NOT_SUPPORTED, f"{name} on sm_{family} returned {status}, expected NotSupported"
        return
    assert status == 0, f"lx_moe_experts_grouped returned CUDA error {status}"
    check(out, case, math)


@pytest.mark.parametrize("T", [16, 1024, 4096])
@pytest.mark.parametrize("name,wtype,math,oldest", [PAIRS[0], PAIRS[4], PAIRS[5], PAIRS[7]])
def test_moe_grouped_praxis_skewed(lx, name, wtype, math, oldest, T):
    # Praxis's experts at prefill sizes with most tokens on a few experts (tiles of every height in use).
    if native_arch(lx) < oldest:
        pytest.skip("not on this GPU")
    L, I, E, top_k = 1024, 512, 128, 4
    case = make_case(T, L, I, E, top_k, wtype, seed=T + math, skew=True)
    status, out = run(lx, case, math, T, L, I, E, top_k)
    assert status == 0, f"lx_moe_experts_grouped returned CUDA error {status}"
    check(out, case, math)


def test_moe_grouped_routing_is_stable(lx):
    # Two calls give the same bits whenever the sum is order-free: with top_k = 1 each output row has one
    # writer, so any difference would come from the sort or the tiling.
    T, L, I, E, top_k = 777, 512, 256, 32, 1
    case = make_case(T, L, I, E, top_k, W_INT8, seed=3)
    _, a = run(lx, case, MATH_INT8, T, L, I, E, top_k)
    _, b = run(lx, case, MATH_INT8, T, L, I, E, top_k)
    assert torch.equal(a, b)
    check(a, case, MATH_INT8)


def test_moe_grouped_empty_and_invalid(lx):
    case = make_case(4, 256, 256, 8, 2, W_BF16, seed=1)
    latent, ids, weights, gu, dn = case
    out = torch.zeros(4, 256, device=DEVICE)
    ws = torch.zeros(workspace_bytes(lx, 4, 256, 256, 8, 2, MATH_BF16), dtype=torch.uint8, device=DEVICE)
    assert call_grouped(lx, latent, ids, weights, gu, dn, MATH_BF16, out, ws, 0, 256, 256, 8, 2) == 0
    assert workspace_bytes(lx, 0, 256, 256, 8, 2, MATH_BF16) == 0
    # I not a multiple of 8, and FP4 math with K not a whole number of its 256-wide K tiles
    assert call_grouped(lx, latent, ids, weights, gu, dn, MATH_BF16, out, ws, 4, 256, 252, 8, 2) == NOT_SUPPORTED
    if BACKEND == "cuda" and native_arch(lx) >= 120:  # the CUDA FP4 path's 256-wide K tiles; Metal takes any K % 16
        g4 = quantize_weight(torch.randn(8, 2 * 128, 256, device=DEVICE), W_NVFP4)
        d4 = quantize_weight(torch.randn(8, 256, 128, device=DEVICE), W_NVFP4)
        ws4 = torch.zeros(workspace_bytes(lx, 4, 256, 128, 8, 2, MATH_FP4), dtype=torch.uint8, device=DEVICE)
        assert call_grouped(lx, latent, ids, weights, g4, d4, MATH_FP4, out, ws4, 4, 256, 128, 8, 2) == NOT_SUPPORTED


@pytest.mark.skipif(BACKEND != "cuda", reason="the CUDA kernel families and their tile variants")
def test_moe_grouped_tuned_variants(lx):
    # Every registered tile variant of every family, picked through the tuner for both stages.
    names = (ctypes.c_char_p * 256)()
    lx.lib.lx_tune_kernels.restype = ctypes.c_int
    count = lx.lib.lx_tune_kernels(names, 256)
    lx.lib.lx_tune_variant_name.restype = ctypes.c_char_p
    families = [names[i].decode() for i in range(min(count, 256)) if names[i].decode().startswith("moe_grouped.")]
    assert len(families) == 10, families
    by_kind = {"bf16": PAIRS[0], "w8i8": PAIRS[1], "w8f8": PAIRS[2], "w4a16": PAIRS[3], "i8": PAIRS[4], "f8": PAIRS[5],
               "w4a8": PAIRS[6], "f4": PAIRS[7]}
    have = native_arch(lx)
    T, L, I, E, top_k = 300, 512, 256, 16, 4
    P = T * top_k
    for fam in families:
        _, kind, sm = fam.split(".")
        name, wtype, math, _ = by_kind[kind]
        fam_arch = int(sm[2:])
        if fam_arch > have:
            continue
        case = make_case(T, L, I, E, top_k, wtype, seed=11, skew=True)
        for v in range(lx.lib.lx_tune_variants(fam.encode())):
            lx.lib.lx_tune_set(fam.encode(), P, 2 * I, L, v)
            lx.lib.lx_tune_set(fam.encode(), P, L, I, v)
            try:
                set_arch(lx, fam_arch if fam_arch < have else 0)
                status, out = run(lx, case, math, T, L, I, E, top_k)
            finally:
                set_arch(lx, 0)
                lx.lib.lx_tune_clear()
            label = f"{fam} variant {lx.lib.lx_tune_variant_name(fam.encode(), v)}"
            assert status == 0, f"{label} returned {status}"
            check(out, case, math)
