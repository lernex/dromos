"""lx_moe_experts_grouped (engine/cuda/moe_grouped.cu) against PyTorch, for every weight format x math
pair and every kernel family (the card's own, and the sm_89 and sm_80 families through
lx_set_arch_override), at Praxis's and Logos's expert shapes and at small and odd ones.

The reference follows the kernel's arithmetic in float64: the latent as the math sees it (bf16, or
quantized per token with the reference quantizer that test_quant_act checks bit for bit), the experts'
weights dequantized from their stored format, the SwiGLU intermediate rounded to bf16, then that
intermediate as the math sees it (quantized per pair row) through the down projection, weighted and summed
per token. What is left is float32 accumulation order, the kernel's exp in silu (CUDA's fast __expf, Metal's
exp), and the bf16 and quantization roundings those can tip, so the bound is relative to the output's
magnitude: 2e-3 under BF16 and INT8 math, 6e-3 under FP4 math (an e2m1 step is a quarter of its block's
scale, so a tipped intermediate moves further), and 8e-3 under FP8 math. Under FP8 the intermediate is
rounded to e4m3 per pair row: one rounding flip of one intermediate moves its token's output row by up to
about half a percent of the output's largest magnitude, and it can happen at any sequence length, since
the chance of a flip grows with the number of rows. Measured on the Metal backend over 16 random cases
(T = 1024 and 4096, skewed routing), the worst case was 5.3e-3, and every case over 2e-3 had a single row
over it. 8e-3 leaves margin over that while still catching a systematic error, which moves every row."""

import ctypes

import pytest
import torch
import torch.nn.functional as F

from conftest import BACKEND, close
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
ARCHS = [0, 89, 80] if BACKEND == "cuda" else [0]
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
    # The GPU families gate the pairs; every other backend runs every pair.
    return lx.lib.lx_arch() if BACKEND == "cuda" else 1000


def workspace_bytes(lx, T, L, I, E, top_k, math):
    fn = lx.lib.lx_moe_grouped_workspace_bytes
    fn.restype = ctypes.c_int64
    return fn(T, L, I, E, top_k, math)


def call_grouped(lx, latent, ids, weights, gu, dn, math, out, ws, T, L, I, E, top_k):
    fn = lx.lib.lx_moe_experts_grouped
    fn.restype = ctypes.c_int
    p = lambda t: ctypes.c_void_p(t.data_ptr())
    status = fn(p(latent), p(ids), p(weights), ctypes.byref(gu.struct), ctypes.byref(dn.struct), math, p(out), p(ws),
                T, L, I, E, top_k, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    return status


def expert(w, e):
    """Expert e of a stacked weight, dequantized to float64 (one expert at a time keeps memory small), with
    its own global when the stack carries one per expert."""
    scale = w.scale[e] if w.scale is not None else None
    globs = getattr(w, "globs", None)
    return dequant_weight(Weight(w.wtype, w.data[e], scale, globs[e] if globs else w.glob))


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
    latent = torch.randn(T, L, device="cuda").to(torch.bfloat16)
    if skew:
        # Most tokens on the same few experts: experts with several tiles of rows next to empty ones.
        hot = torch.randperm(E, device="cuda")[: max(top_k, 3)]
        noise = torch.rand(T, E, device="cuda")
        noise[:, hot] += 10.0 * (torch.rand(T, 1, device="cuda") < 0.9)
        ids = noise.argsort(-1, descending=True)[:, :top_k]
    else:
        ids = torch.rand(T, E, device="cuda").argsort(-1)[:, :top_k]
    ids = ids.to(torch.int32).contiguous()
    weights = torch.softmax(torch.randn(T, top_k, device="cuda"), -1).contiguous()
    gu = quantize_weight(torch.randn(E, 2 * I, L, device="cuda") / L**0.5, wtype)
    dn = quantize_weight(torch.randn(E, L, I, device="cuda") / I**0.5, wtype)
    return latent, ids, weights, gu, dn


def run(lx, case, math, T, L, I, E, top_k):
    latent, ids, weights, gu, dn = case
    out = torch.full((T, L), float("nan"), device="cuda")
    ws = torch.full((max(workspace_bytes(lx, T, L, I, E, top_k, math), 1),), 0xFF, dtype=torch.uint8, device="cuda")
    return call_grouped(lx, latent, ids, weights, gu, dn, math, out, ws, T, L, I, E, top_k), out


# Relative to the output's largest magnitude; the reasons are in the module docstring.
TOLERANCE = {MATH_BF16: 2e-3, MATH_INT8: 2e-3, MATH_FP8: 8e-3, MATH_FP4: 6e-3}


def check(out, case, math):
    latent, ids, weights, gu, dn = case
    want = reference(latent, ids, weights, gu, dn, math)
    tol = TOLERANCE[math] * want.abs().max().item()
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


def per_expert_nvfp4(E, N, K, seed):
    """An NVFP4 stack quantized one expert at a time, so each expert has its own global (expert_global), as
    checkpoints carry them and the loader's conversion makes them. The experts' magnitudes differ by up to 46x,
    so a kernel that applies one expert's global to the others is far off."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    parts = [quantize_weight((torch.randn(N, K, generator=g) * (1 + 3 * e) / K**0.5).cuda(), W_NVFP4) for e in range(E)]
    w = Weight(W_NVFP4, torch.stack([p.data for p in parts]), torch.stack([p.scale for p in parts]), parts[0].glob)
    w.globs = [p.glob for p in parts]
    w.expert_globals = torch.tensor(w.globs, dtype=torch.float32, device="cuda")
    w.struct.expert_global = w.expert_globals.data_ptr()
    return w


@pytest.mark.parametrize("name,wtype,math,oldest", [p for p in PAIRS if p[1] == W_NVFP4])
def test_moe_grouped_nvfp4_expert_globals(lx, name, wtype, math, oldest):
    # Every NVFP4 kind reads each expert's own global, as lx_moe_experts_decode does.
    if native_arch(lx) < oldest:
        pytest.skip("not on this GPU")
    T, L, I, E, top_k = 37, 512, 256, 16, 3
    latent, ids, weights, _, _ = make_case(T, L, I, E, top_k, W_BF16, seed=17)
    case = (latent, ids, weights, per_expert_nvfp4(E, 2 * I, L, 1), per_expert_nvfp4(E, L, I, 2))
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
    out = torch.zeros(4, 256, device="cuda")
    ws = torch.zeros(workspace_bytes(lx, 4, 256, 256, 8, 2, MATH_BF16), dtype=torch.uint8, device="cuda")
    assert call_grouped(lx, latent, ids, weights, gu, dn, MATH_BF16, out, ws, 0, 256, 256, 8, 2) == 0
    assert workspace_bytes(lx, 0, 256, 256, 8, 2, MATH_BF16) == 0
    # I not a multiple of 8, and FP4 math with K not a whole number of its 256-wide K tiles
    if BACKEND != "cuda":
        return  # the tile-shape limits below are the CUDA kernels'
    assert call_grouped(lx, latent, ids, weights, gu, dn, MATH_BF16, out, ws, 4, 256, 252, 8, 2) == NOT_SUPPORTED
    if native_arch(lx) >= 120:
        g4 = quantize_weight(torch.randn(8, 2 * 128, 256, device="cuda"), W_NVFP4)
        d4 = quantize_weight(torch.randn(8, 256, 128, device="cuda"), W_NVFP4)
        ws4 = torch.zeros(workspace_bytes(lx, 4, 256, 128, 8, 2, MATH_FP4), dtype=torch.uint8, device="cuda")
        assert call_grouped(lx, latent, ids, weights, g4, d4, MATH_FP4, out, ws4, 4, 256, 128, 8, 2) == NOT_SUPPORTED


def test_moe_grouped_tuned_variants(lx):
    if BACKEND != "cuda":
        pytest.skip("the tile variants are the CUDA kernels'")
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


@pytest.mark.parametrize("name,wtype,math,oldest", PAIRS)
def test_moe_grouped_rows_do_not_depend_on_the_batch(lx, name, wtype, math, oldest):
    # Batch invariance on the decode rows (T <= 8, where a speculative verify step runs): a token's row is the
    # same bits whatever batch of T (1 to 8) it runs in, so a verify step matches the one-token steps.
    if native_arch(lx) < oldest:
        pytest.skip("not on this GPU")
    L, I, E, top_k = 1024, 512, 32, 4
    latent, ids, weights, gu, dn = make_case(8, L, I, E, top_k, wtype, seed=77 + math)
    alone = []
    for t in range(8):
        sub = (latent[t : t + 1].contiguous(), ids[t : t + 1].contiguous(), weights[t : t + 1].contiguous(), gu, dn)
        status, out = run(lx, sub, math, 1, L, I, E, top_k)
        assert status == 0, f"lx_moe_experts_grouped returned CUDA error {status}"
        alone.append(out)
    alone = torch.cat(alone)
    for T in range(1, 9):
        sub = (latent[:T].contiguous(), ids[:T].contiguous(), weights[:T].contiguous(), gu, dn)
        status, out = run(lx, sub, math, T, L, I, E, top_k)
        assert status == 0, f"lx_moe_experts_grouped returned CUDA error {status}"
        for t in range(T):
            assert torch.equal(out[t], alone[t]), f"{name}: token {t} of a T={T} batch differs from the token alone"


def native_family(lx, name):
    """The variant family that runs pair `name` on this card, as moe_grouped.cu dispatches it."""
    kind = {"bf16": "bf16", "w8a16_int8": "w8i8", "w8a16_fp8": "w8f8", "w4a16": "w4a16", "int8": "i8", "fp8": "f8",
            "w4a8": "w4a8", "fp4": "f4"}[name]
    names = (ctypes.c_char_p * 256)()
    lx.lib.lx_tune_kernels.restype = ctypes.c_int
    count = lx.lib.lx_tune_kernels(names, 256)
    fams = [names[i].decode() for i in range(min(count, 256)) if names[i].decode().startswith(f"moe_grouped.{kind}.sm")]
    have = native_arch(lx)
    cands = [f for f in fams if int(f.split(".sm")[1]) <= have]
    if not cands:
        return None  # a backend with no tuner families for this kind (Metal) runs its one kernel, unforced
    return max(cands, key=lambda f: int(f.split(".sm")[1])).encode()


@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("name,wtype,math,oldest", PAIRS)
def test_moe_grouped_repeats_bit_for_bit(lx, name, wtype, math, oldest, skew):
    # The engine rule, run to run: five calls at a prefill size with Praxis's widest routing (top 8) give the
    # same bits, each from a fresh workspace full of garbage. Thousands of tiles write a token's pair rows in
    # whatever order the GPU runs them; the sum over them must not show it.
    if native_arch(lx) < oldest:
        pytest.skip("not on this GPU")
    T, L, I, E, top_k = 1024, 1024, 512, 128, 8
    case = make_case(T, L, I, E, top_k, wtype, seed=91 + math, skew=skew)
    status, first = run(lx, case, math, T, L, I, E, top_k)
    assert status == 0, f"lx_moe_experts_grouped returned CUDA error {status}"
    for _ in range(4):
        status, out = run(lx, case, math, T, L, I, E, top_k)
        assert status == 0, f"lx_moe_experts_grouped returned CUDA error {status}"
        assert torch.equal(out, first), f"{name}: a repeated call gave different bits"


@pytest.mark.parametrize("name,wtype,math,oldest", PAIRS)
def test_moe_grouped_rows_do_not_depend_on_a_prefill_batch(lx, name, wtype, math, oldest):
    # The engine rule at prefill sizes: a token's row is the same bits whether it shares the launch with 63 or
    # with 511 other tokens, and whichever tile variant of its family runs the 64 (another tile height and
    # other tile tables; every tile adds the K steps in one order).
    if native_arch(lx) < oldest:
        pytest.skip("not on this GPU")
    T, L, I, E, top_k = 512, 1024, 512, 128, 4
    case = make_case(T, L, I, E, top_k, wtype, seed=123 + math)
    status, big = run(lx, case, math, T, L, I, E, top_k)
    assert status == 0, f"lx_moe_experts_grouped returned CUDA error {status}"
    small = tuple(x[:64].contiguous() for x in case[:3]) + case[3:]
    fam = native_family(lx, name)
    if fam is None:
        status, out = run(lx, small, math, 64, L, I, E, top_k)
        assert status == 0, f"{name} returned CUDA error {status}"
        assert torch.equal(out, big[:64]), f"{name}: the first 64 tokens differ from their rows in a batch of 512"
        return
    lx.lib.lx_tune_variant_name.restype = ctypes.c_char_p
    for v in range(lx.lib.lx_tune_variants(fam)):
        lx.lib.lx_tune_force(fam, 64 * top_k, 2 * I, L, v)
        lx.lib.lx_tune_force(fam, 64 * top_k, L, I, v)
        try:
            status, out = run(lx, small, math, 64, L, I, E, top_k)
        finally:
            lx.lib.lx_tune_clear()
        label = f"{name}, variant {lx.lib.lx_tune_variant_name(fam, v).decode()}"
        assert status == 0, f"{label} returned CUDA error {status}"
        assert torch.equal(out, big[:64]), f"{label}: the first 64 tokens differ from their rows in a batch of 512"


@pytest.mark.parametrize("name,wtype,math,oldest", [PAIRS[0], PAIRS[4], PAIRS[7]])
def test_moe_grouped_expert_chosen_twice(lx, name, wtype, math, oldest):
    # A token routed to the same expert in two slots: its two rows sit side by side in one expert's run, often
    # in one tile, and both count towards the token's sum like any other pair.
    if native_arch(lx) < oldest:
        pytest.skip("not on this GPU")
    T, L, I, E, top_k = 300, 512, 256, 16, 4
    latent, ids, weights, gu, dn = make_case(T, L, I, E, top_k, wtype, seed=7)
    ids[::3, 2] = ids[::3, 0]
    case = (latent, ids.contiguous(), weights, gu, dn)
    status, a = run(lx, case, math, T, L, I, E, top_k)
    assert status == 0, f"lx_moe_experts_grouped returned CUDA error {status}"
    status, b = run(lx, case, math, T, L, I, E, top_k)
    assert torch.equal(a, b)
    check(a, case, math)
