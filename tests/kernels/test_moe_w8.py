"""The latent mixture of experts with int8 expert weights (engine/cuda/moe_w8.cu) against PyTorch.

Weights are quantized here with the spec's rounding: per-row absmax / 127 (1 for an all-zero row),
round half to even, clamp to [-127, 127]. The reference dequantizes them (q * scale) and computes the
experts in float64, rounding the SwiGLU intermediate to bf16 as the kernel does, so the only
difference left is float32 accumulation. Compared to 1e-3 of the output's magnitude.
"""

import ctypes

import pytest
import torch
import torch.nn.functional as F

from conftest import DEVICE, close

METIS_E = [128, 192]
METIS_T = [1, 2, 4, 8, 64, 300]
METIS_L, METIS_I = 1024, 512
TOP_K = 4


def quantize_w8(w, chunk=16):
    """Per-row symmetric int8 of w [..., rows, cols]. Returns (q int8 like w, scale f32 [..., rows])."""
    qs, ss = [], []
    for start in range(0, w.shape[0], chunk):
        wf = w[start : start + chunk].float()
        absmax = wf.abs().amax(dim=-1)
        scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
        q = torch.round(wf / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        qs.append(q)
        ss.append(scale)
    return torch.cat(qs).contiguous(), torch.cat(ss).contiguous()


def make_w8_experts(T, E, L, I, top_k, ids=None):
    """Random bf16 experts, quantized. Routing is distinct experts per token unless ids is given."""
    latent = torch.randn(T, L, device=DEVICE).to(torch.bfloat16).contiguous()
    gate_up = (torch.randn(E, 2 * I, L, device=DEVICE) / L**0.5).to(torch.bfloat16)
    down = (torch.randn(E, L, I, device=DEVICE) / I**0.5).to(torch.bfloat16)
    q_gu, s_gu = quantize_w8(gate_up)
    q_dn, s_dn = quantize_w8(down)
    if ids is None:
        ids = torch.rand(T, E, device=DEVICE).argsort(-1)[:, :top_k]
    ids = ids.to(torch.int32).contiguous()
    weights = torch.softmax(torch.randn(T, top_k, device=DEVICE), -1).contiguous()
    return latent, ids, weights, q_gu, s_gu, q_dn, s_dn


def experts_reference_w8(latent, ids, weights, q_gu, s_gu, q_dn, s_dn):
    """Float64 experts on the dequantized weights; the intermediate is rounded to bf16."""
    I = q_gu.shape[1] // 2
    x = latent.double()
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float64, device=x.device)
    for e in ids.unique().tolist():
        gu = q_gu[e].double() * s_gu[e].double()[:, None]  # [2I, L]
        dn = q_dn[e].double() * s_dn[e].double()[:, None]  # [L, I]
        for k in range(ids.shape[1]):
            rows = (ids[:, k] == e).nonzero().squeeze(1)
            if rows.numel() == 0:
                continue
            xe = x[rows]
            mid = (F.silu(xe @ gu[:I].T) * (xe @ gu[I:].T)).to(torch.bfloat16).double()
            out.index_add_(0, rows, weights[rows, k].double()[:, None] * (mid @ dn.T))
    return out


def workspace_bytes(lx, T, I, top_k):
    fn = lx.lib.lx_moe_workspace_bytes
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    return fn(T, I, top_k)


def run_w8(lx, latent, ids, weights, q_gu, s_gu, q_dn, s_dn):
    """Runs the kernel in a workspace of exactly the reported size, filled with NaN bytes first, so
    the result cannot depend on stale scratch and the call cannot need more than was reported."""
    T, L = latent.shape
    I = q_gu.shape[1] // 2
    top_k = ids.shape[1]
    nbytes = workspace_bytes(lx, T, I, top_k)
    assert nbytes >= 2 * T * top_k * I, "workspace smaller than the bf16 intermediates it holds"
    ws = torch.full((nbytes,), 255, dtype=torch.uint8, device=DEVICE)
    # Written, not accumulated: the output starts as NaN and must come out fully written.
    out = torch.full((T, L), float("nan"), device=DEVICE)
    lx("moe_experts_w8", latent, ids, weights, q_gu, s_gu, q_dn, s_dn, out, ws, T, L, I, top_k)
    return out


def check_against_reference(lx, latent, ids, weights, q_gu, s_gu, q_dn, s_dn):
    out = run_w8(lx, latent, ids, weights, q_gu, s_gu, q_dn, s_dn)
    want = experts_reference_w8(latent, ids, weights, q_gu, s_gu, q_dn, s_dn)
    scale = want.abs().max().item()
    close(out, want.float(), atol=1e-3 * scale, rtol=1e-3)
    return out


@pytest.mark.parametrize("E", METIS_E)
@pytest.mark.parametrize("T", METIS_T)
def test_moe_w8_metis(lx, E, T):
    torch.manual_seed(E * 1000 + T)
    args = make_w8_experts(T, E, METIS_L, METIS_I, TOP_K)
    check_against_reference(lx, *args)


@pytest.mark.parametrize(
    "T,E,L,I,top_k",
    [
        (3, 5, 24, 16, 2),  # L scalar stage 1 (24 % 16 != 0), I vector stage 2 with a partial chunk
        (7, 9, 101, 33, 3),  # odd L and I: both scalar
        (5, 6, 64, 48, 4),  # vector paths, a row shorter than one warp chunk
        (4, 4, 1032, 36, 4),  # L a multiple of 8 but not 16; I not a multiple of 16
        (1, 1, 8, 8, 1),  # one expert, one token
        (2, 8, 1536, 1024, 4),  # three chunks per row in stage 1, two in stage 2
        (2, 4, 2048, 512, 2),  # the largest vector row (four chunks)
        (5, 16, 1040, 512, 4),  # L % 16 == 0 with a partial last chunk
        (3, 12, 1024, 512, 8),  # top_k at the register-path limit
        (12, 6, 64, 48, 4),  # more than 32 pairs: the sharing path, vector rows
        (12, 9, 101, 33, 3),  # more than 32 pairs: the sharing path, scalar rows
        (6, 12, 1024, 512, 8),  # more than 32 pairs at top_k 8: chains of several tokens
    ],
)
def test_moe_w8_shapes(lx, T, E, L, I, top_k):
    torch.manual_seed(T * 100 + E * 10 + L + I)
    args = make_w8_experts(T, E, L, I, top_k)
    check_against_reference(lx, *args)


def test_moe_w8_repeated_experts(lx):
    # Tokens share a small pool of experts, so most expert reads are shared across pairs.
    torch.manual_seed(21)
    T, E = 16, 128
    pool = torch.randperm(E, device=DEVICE)[:6]
    ids = pool[torch.rand(T, 6, device=DEVICE).argsort(-1)[:, :TOP_K]]
    check_against_reference(lx, *make_w8_experts(T, E, METIS_L, METIS_I, TOP_K, ids=ids))


def test_moe_w8_identical_routing(lx):
    # Every token picks the same four experts: each expert's leader pair covers all T tokens.
    torch.manual_seed(22)
    T, E = 300, 128
    ids = torch.tensor([[5, 17, 60, 99]], device=DEVICE).expand(T, TOP_K)
    check_against_reference(lx, *make_w8_experts(T, E, METIS_L, METIS_I, TOP_K, ids=ids))


def test_moe_w8_deterministic(lx):
    # No atomics: the same inputs give bit-identical outputs, across runs and batch sizes.
    torch.manual_seed(11)
    T, E = 300, 128
    latent, ids, weights, q_gu, s_gu, q_dn, s_dn = make_w8_experts(T, E, METIS_L, METIS_I, TOP_K)
    first = run_w8(lx, latent, ids, weights, q_gu, s_gu, q_dn, s_dn)
    second = run_w8(lx, latent, ids, weights, q_gu, s_gu, q_dn, s_dn)
    assert torch.equal(first, second)
    # A token's output does not depend on the other tokens in the call.
    head = run_w8(lx, latent[:8], ids[:8], weights[:8], q_gu, s_gu, q_dn, s_dn)
    assert torch.equal(head, first[:8])


def test_moe_w8_workspace_contract(lx):
    # The reported size covers the intermediates for every shape, and it is what the kernel needs.
    for T, top_k, I in [(1, 4, 512), (8, 4, 512), (300, 4, 512), (3, 8, 33)]:
        nbytes = workspace_bytes(lx, T, I, top_k)
        assert nbytes >= 2 * T * top_k * I
        assert nbytes % 256 == 0
    # Intermediates first, then two int32 link arrays of one entry per pair (the sharing path).
    assert workspace_bytes(lx, 1, 512, 4) >= 4096 + 2 * 4 * 4
    assert workspace_bytes(lx, 1, 512, 4) % 256 == 0
