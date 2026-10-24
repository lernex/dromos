"""Few-token latent experts on every weight format (engine/cuda/moe_decode.cu, lx_moe_experts_decode).

The contract of lx_moe_experts (kernels.h): out[t] = sum_k w[t, k] * down[e] @ (silu(gate_up[e][:I] @ x)
* (gate_up[e][I:] @ x)), with the intermediate rounded to bf16. The stacks are random in their format
(formats.py); the reference is float64 on the exact values the bytes stand for, and it rounds the
intermediate to bf16 the way the kernel does, so only float32 accumulation is left to differ.
"""

import ctypes

import pytest
import torch
import torch.nn.functional as F

from conftest import DEVICE, close
from formats import FORMATS, W_BF16, W_FP8, W_INT8, W_NVFP4, stack_weight

METIS = (1024, 512, 4)  # L, I, top_k of Praxis


def workspace_bytes(lx, T, I, top_k):
    fn = lx.lib.lx_moe_decode_workspace_bytes
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    return fn(T, I, top_k)


def experts_reference(latent, ids, weights, gu, dn):
    """float64 experts on the stacks' dense values (gu [E, 2I, L], dn [E, L, I]); intermediate in bf16."""
    I = gu.shape[1] // 2
    x = latent.double()
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float64, device=x.device)
    for e in ids.unique().tolist():
        for k in range(ids.shape[1]):
            rows = (ids[:, k] == e).nonzero().squeeze(1)
            if rows.numel() == 0:
                continue
            xe = x[rows]
            mid = (F.silu(xe @ gu[e, :I].T) * (xe @ gu[e, I:].T)).to(torch.bfloat16).double()
            out.index_add_(0, rows, weights[rows, k].double()[:, None] * (mid @ dn[e].T))
    return out


def run_decode(lx, latent, ids, weights, gu_w, dn_w, I, top_k):
    """The kernel in a workspace of exactly the reported size, filled with NaN bytes first."""
    T, L = latent.shape
    nbytes = workspace_bytes(lx, T, I, top_k)
    assert nbytes >= 2 * T * top_k * I, "workspace smaller than the bf16 intermediates it holds"
    ws = torch.full((nbytes,), 255, dtype=torch.uint8, device=DEVICE)
    out = torch.full((T, L), float("nan"), device=DEVICE)
    gu_lx, dn_lx = ctypes.pointer(gu_w.lx()), ctypes.pointer(dn_w.lx())
    lx("moe_experts_decode", latent, ids, weights, gu_lx, dn_lx, out, ws, T, L, I, top_k)
    return out


def check(lx, latent, ids, weights, gu_w, dn_w, I, top_k):
    out = run_decode(lx, latent, ids, weights, gu_w, dn_w, I, top_k)
    want = experts_reference(latent, ids, weights, gu_w.dense, dn_w.dense)
    scale = want.abs().max().item()
    close(out, want.float(), atol=1e-3 * scale, rtol=1e-3)
    return out


def make_inputs(T, E, L, I, top_k, seed, pool=None):
    torch.manual_seed(seed)
    latent = torch.randn(T, L, device=DEVICE).to(torch.bfloat16).contiguous()
    if pool is None:
        ids = torch.rand(T, E, device=DEVICE).argsort(-1)[:, :top_k]
    else:
        ids = pool[torch.rand(T, len(pool), device=DEVICE).argsort(-1)[:, :top_k]]
    ids = ids.to(torch.int32).contiguous()
    weights = torch.softmax(torch.randn(T, top_k, device=DEVICE), -1).contiguous()
    return latent, ids, weights


@pytest.mark.parametrize("fmt", list(FORMATS))
@pytest.mark.parametrize("T", [1, 2, 4, 8])
def test_metis_shapes(lx, T, fmt):
    L, I, top_k = METIS
    E = 32
    gu = stack_weight(FORMATS[fmt], E, 2 * I, L, seed=T + 1)
    dn = stack_weight(FORMATS[fmt], E, L, I, seed=T + 2)
    latent, ids, weights = make_inputs(T, E, L, I, top_k, seed=T * 13)
    check(lx, latent, ids, weights, gu, dn, I, top_k)


@pytest.mark.parametrize(
    "T,E,L,I,top_k",
    [
        (1, 1, 16, 16, 1),
        (3, 5, 48, 32, 2),
        (5, 9, 64, 48, 3),
        (8, 8, 32, 64, 8),  # top_k at the register limit
        (2, 6, 1024, 16, 4),  # a short intermediate: the stage 2 rows are one lane word each
    ],
)
@pytest.mark.parametrize("fmt", list(FORMATS))
def test_odd_shapes(lx, T, E, L, I, top_k, fmt):
    gu = stack_weight(FORMATS[fmt], E, 2 * I, L, seed=E + 3)
    dn = stack_weight(FORMATS[fmt], E, L, I, seed=E + 4)
    latent, ids, weights = make_inputs(T, E, L, I, top_k, seed=T + E)
    check(lx, latent, ids, weights, gu, dn, I, top_k)


@pytest.mark.parametrize("gu_fmt,dn_fmt", [("nvfp4", "fp8"), ("fp8", "nvfp4"), ("int8", "bf16"), ("bf16", "nvfp4")])
def test_mixed_formats_per_stack(lx, gu_fmt, dn_fmt):
    # Gate/up and down are separate stacks, each in its own format.
    T, E = 4, 16
    L, I, top_k = METIS
    gu = stack_weight(FORMATS[gu_fmt], E, 2 * I, L, seed=21)
    dn = stack_weight(FORMATS[dn_fmt], E, L, I, seed=22)
    latent, ids, weights = make_inputs(T, E, L, I, top_k, seed=23)
    check(lx, latent, ids, weights, gu, dn, I, top_k)


def test_repeated_experts(lx):
    # Tokens share a small pool of experts: most expert rows are read by several pairs.
    T, E = 8, 128
    L, I, top_k = METIS
    gu = stack_weight(W_NVFP4, E, 2 * I, L, seed=31)
    dn = stack_weight(W_NVFP4, E, L, I, seed=32)
    pool = torch.randperm(E, device=DEVICE)[:6]
    latent, ids, weights = make_inputs(T, E, L, I, top_k, seed=33, pool=pool)
    check(lx, latent, ids, weights, gu, dn, I, top_k)


@pytest.mark.parametrize("fmt", list(FORMATS))
def test_deterministic_and_token_independent(lx, fmt):
    # No atomics: the same inputs give bit-identical outputs, and a token's row does not depend on the others.
    T, E = 8, 32
    L, I, top_k = METIS
    gu = stack_weight(FORMATS[fmt], E, 2 * I, L, seed=41)
    dn = stack_weight(FORMATS[fmt], E, L, I, seed=42)
    latent, ids, weights = make_inputs(T, E, L, I, top_k, seed=43)
    first = run_decode(lx, latent, ids, weights, gu, dn, I, top_k)
    second = run_decode(lx, latent, ids, weights, gu, dn, I, top_k)
    assert torch.equal(first, second)
    head = run_decode(lx, latent[:1], ids[:1], weights[:1], gu, dn, I, top_k)
    assert torch.equal(head, first[:1])


def test_rejects_more_than_eight_tokens(lx):
    # The decode kernel is for a few tokens; larger steps belong to the grouped kernel.
    T, E = 9, 8
    L, I, top_k = METIS
    gu = stack_weight(W_BF16, E, 2 * I, L, seed=51)
    dn = stack_weight(W_BF16, E, L, I, seed=52)
    latent, ids, weights = make_inputs(T, E, L, I, top_k, seed=53)
    nbytes = workspace_bytes(lx, T, I, top_k)
    ws = torch.empty(nbytes, dtype=torch.uint8, device=DEVICE)
    out = torch.empty(T, L, device=DEVICE)
    fn = lx.lib.lx_moe_experts_decode
    fn.restype = ctypes.c_int
    status = fn(
        ctypes.c_void_p(latent.data_ptr()), ctypes.c_void_p(ids.data_ptr()), ctypes.c_void_p(weights.data_ptr()),
        ctypes.pointer(gu.lx()), ctypes.pointer(dn.lx()), ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(ws.data_ptr()), T, L, I, top_k, lx.stream,
    )
    assert status != 0  # rejected before any launch


def test_global_is_per_expert(lx):
    # Each expert reads its own NVFP4 global. The down global enters linearly: doubling expert 1's down
    # global doubles the output exactly, and expert 3, which has weight 0, changes nothing.
    T, E = 1, 4
    L, I, top_k = METIS
    gu = stack_weight(W_NVFP4, E, 2 * I, L, seed=61)
    dn = stack_weight(W_NVFP4, E, L, I, seed=62)
    latent, _, _ = make_inputs(T, E, L, I, 2, seed=63)
    ids = torch.tensor([[1, 3]], dtype=torch.int32, device=DEVICE)
    weights = torch.tensor([[1.0, 0.0]], device=DEVICE)
    base = run_decode(lx, latent, ids, weights, gu, dn, I, 2)
    dn.expert_global[3] *= 2.0
    assert torch.equal(run_decode(lx, latent, ids, weights, gu, dn, I, 2), base)
    dn.expert_global[1] *= 2.0
    assert torch.equal(run_decode(lx, latent, ids, weights, gu, dn, I, 2), base * 2.0)
