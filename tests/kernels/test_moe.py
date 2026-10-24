"""The latent mixture of experts kernels (engine/cuda/moe.cu) against PyTorch references.

Routing: the expert ids are compared exactly. The selection is made on float32 biased logits, the
same IEEE add the kernel does, so a tie is a tie on both sides and resolves to the lower index.
Weights, entropy and confidence are compared to 1e-5 against float64.

Experts: compared to a float64 reference that rounds the SwiGLU intermediate to bf16 as the kernel
does, to 1e-3 of the output's magnitude. Summary: compared to float64 to 1e-5.
"""

import pytest
import torch
import torch.nn.functional as F

from conftest import close

METIS_E = [128, 192]
TOP_K = 4


def route_reference(logits, bias, top_k):
    """Routing in float64, with the selection done on float32 biased logits as the kernel does."""
    biased = logits.float() if bias is None else logits.float() + bias.float()
    # Sorting -biased ascending (stable) gives descending biased logits with ties to the lower index.
    ids = torch.sort(-biased, dim=-1, stable=True).indices[:, :top_k]
    wide = logits.double()
    weights = torch.softmax(torch.gather(wide, -1, ids), dim=-1)
    p = torch.softmax(wide, dim=-1)
    entropy = -(p * torch.log(p.clamp_min(1e-9))).sum(-1)
    confidence = p.max(-1).values
    return ids, weights, entropy, confidence


def run_route(lx, logits, bias, top_k):
    T, E = logits.shape
    ids = torch.empty(T, top_k, dtype=torch.int32, device="cuda")
    weights = torch.empty(T, top_k, dtype=torch.float32, device="cuda")
    entropy = torch.empty(T, dtype=torch.float32, device="cuda")
    confidence = torch.empty(T, dtype=torch.float32, device="cuda")
    lx("moe_route", logits, bias, ids, weights, entropy, confidence, T, E, top_k)
    return ids, weights, entropy, confidence


def check_route(lx, logits, bias, top_k):
    T, E = logits.shape
    # The random cases should not depend on how a near tie resolves: the boundary between the
    # picks and the first non-pick must be clear of ties. The reference is exact either way.
    biased = logits if bias is None else logits + bias
    desc = torch.sort(biased, dim=-1, descending=True).values[:, : top_k + 1]
    if top_k < E:
        gap = (desc[:, :-1] - desc[:, 1:]).min().item()
        assert gap > 1e-5, f"near tie in the test logits (gap {gap:.3e}), pick another seed"
    ids, weights, entropy, confidence = run_route(lx, logits, bias, top_k)
    want_ids, want_w, want_h, want_c = route_reference(logits, bias, top_k)
    assert torch.equal(ids.long(), want_ids), "expert ids differ from the reference"
    close(weights, want_w, atol=1e-5)
    close(entropy, want_h, atol=1e-5)
    close(confidence, want_c, atol=1e-5)


@pytest.mark.parametrize("E,T", [(128, 1), (128, 7), (128, 300), (192, 1), (192, 7), (192, 300)])
def test_moe_route_metis(lx, E, T):
    torch.manual_seed(E * 1000 + T)
    logits = (torch.randn(T, E, device="cuda") * 2).float().contiguous()
    bias = (torch.randn(E, device="cuda") * 0.5).float().contiguous()
    check_route(lx, logits, bias, TOP_K)


@pytest.mark.parametrize("E,T,top_k", [(5, 3, 2), (37, 7, 3), (7, 2, 7), (64, 1, 1)])
def test_moe_route_small_odd(lx, E, T, top_k):
    torch.manual_seed(E * 100 + T * 10 + top_k)
    logits = (torch.randn(T, E, device="cuda") * 2).float().contiguous()
    bias = (torch.randn(E, device="cuda") * 0.5).float().contiguous()
    check_route(lx, logits, bias, top_k)


def test_moe_route_without_bias(lx):
    # A null selection_bias means zero bias: the picks are the plain top-k of the logits.
    torch.manual_seed(7)
    logits = (torch.randn(33, 128, device="cuda") * 2).float().contiguous()
    check_route(lx, logits, None, TOP_K)


def test_moe_route_ties_resolve_to_lower_index(lx):
    # Three experts share the top logit: the picks are the lowest-numbered of them, and the
    # weights over the three equal logits are uniform.
    logits = torch.tensor([[0.5, 2.0, 2.0, -1.0, 2.0, 0.0, 1.0, 0.0]], device="cuda")
    ids, weights, entropy, confidence = run_route(lx, logits, None, 3)
    assert ids.tolist() == [[1, 2, 4]]
    close(weights, torch.full((1, 3), 1.0 / 3.0, device="cuda"), atol=1e-6)
    _, _, want_h, want_c = route_reference(logits, None, 3)
    close(entropy, want_h, atol=1e-5)
    close(confidence, want_c, atol=1e-5)


def test_moe_route_ties_from_bias(lx):
    # The bias makes three biased logits equal (1.0). The picks follow the biased order, but the
    # weights come from the unbiased logits of the picks: softmax([1.0, 0.0]).
    logits = torch.tensor([[1.0, 0.0, 0.5, -2.0, -3.0]], device="cuda")
    bias = torch.tensor([0.0, 1.0, 0.5, 0.0, 0.0], device="cuda")
    ids, weights, entropy, confidence = run_route(lx, logits, bias, 2)
    assert ids.tolist() == [[0, 1]]
    close(weights, torch.softmax(torch.tensor([[1.0, 0.0]], device="cuda"), -1), atol=1e-6)
    _, _, want_h, want_c = route_reference(logits, bias, 2)
    close(entropy, want_h, atol=1e-5)
    close(confidence, want_c, atol=1e-5)


def experts_reference(latent, ids, weights, gate_up, down):
    """Float64 experts. The intermediate is rounded to bf16 before the down projection."""
    I = gate_up.shape[1] // 2
    x = latent.double()
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float64, device=x.device)
    for e in ids.unique().tolist():
        gu = gate_up[e].double()  # [2I, L]
        dn = down[e].double()  # [L, I]
        for k in range(ids.shape[1]):
            rows = (ids[:, k] == e).nonzero().squeeze(1)
            if rows.numel() == 0:
                continue
            xe = x[rows]
            mid = (F.silu(xe @ gu[:I].T) * (xe @ gu[I:].T)).to(torch.bfloat16).double()
            out.index_add_(0, rows, weights[rows, k].double()[:, None] * (mid @ dn.T))
    return out


def make_experts(T, E, L, I, top_k):
    latent = torch.randn(T, L, device="cuda").to(torch.bfloat16).contiguous()
    gate_up = (torch.randn(E, 2 * I, L, device="cuda") / L**0.5).to(torch.bfloat16).contiguous()
    down = (torch.randn(E, L, I, device="cuda") / I**0.5).to(torch.bfloat16).contiguous()
    # Distinct experts per token, as routing produces.
    ids = torch.rand(T, E, device="cuda").argsort(-1)[:, :top_k].to(torch.int32).contiguous()
    weights = torch.softmax(torch.randn(T, top_k, device="cuda"), -1).contiguous()
    return latent, ids, weights, gate_up, down


@pytest.mark.parametrize("E", METIS_E)
@pytest.mark.parametrize("T", [1, 7, 300])
def test_moe_experts_metis(lx, E, T):
    torch.manual_seed(E * 1000 + T)
    L, I = 1024, 512
    latent, ids, weights, gate_up, down = make_experts(T, E, L, I, TOP_K)
    # Written, not accumulated: the output starts as NaN and must come out fully written.
    out = torch.full((T, L), float("nan"), device="cuda")
    lx("moe_experts", latent, ids, weights, gate_up, down, out, T, L, I, TOP_K)
    want = experts_reference(latent, ids, weights, gate_up, down)
    scale = want.abs().max().item()
    close(out, want.float(), atol=1e-3 * scale, rtol=1e-3)


@pytest.mark.parametrize(
    "T,E,L,I,top_k",
    [
        (3, 5, 24, 16, 2),  # L and I not multiples of 8: scalar dot products
        (7, 9, 101, 33, 3),  # odd L and I
        (5, 6, 64, 48, 4),  # vector path, rows shorter than a warp step
        (4, 4, 1032, 36, 4),  # L not a multiple of the warp step; I not a multiple of 8 rows
        (1, 1, 8, 8, 1),  # one expert, one token
    ],
)
def test_moe_experts_small_odd(lx, T, E, L, I, top_k):
    torch.manual_seed(T * 100 + E * 10 + L + I)
    latent, ids, weights, gate_up, down = make_experts(T, E, L, I, top_k)
    out = torch.full((T, L), float("nan"), device="cuda")
    lx("moe_experts", latent, ids, weights, gate_up, down, out, T, L, I, top_k)
    want = experts_reference(latent, ids, weights, gate_up, down)
    scale = want.abs().max().item()
    close(out, want.float(), atol=1e-3 * scale, rtol=1e-3)


def test_moe_experts_deterministic(lx):
    # No atomics: the same inputs give bit-identical outputs.
    torch.manual_seed(11)
    T, E, L, I = 300, 128, 1024, 512
    latent, ids, weights, gate_up, down = make_experts(T, E, L, I, TOP_K)
    first = torch.empty(T, L, device="cuda")
    second = torch.empty(T, L, device="cuda")
    lx("moe_experts", latent, ids, weights, gate_up, down, first, T, L, I, TOP_K)
    lx("moe_experts", latent, ids, weights, gate_up, down, second, T, L, I, TOP_K)
    assert torch.equal(first, second)


@pytest.mark.parametrize("E", METIS_E)
@pytest.mark.parametrize("T", [1, 7, 300])
def test_moe_summary_metis(lx, E, T):
    torch.manual_seed(E * 1000 + T)
    R = 256
    emb = torch.randn(E, R, device="cuda").to(torch.bfloat16).contiguous()
    ids = torch.rand(T, E, device="cuda").argsort(-1)[:, :TOP_K].to(torch.int32).contiguous()
    weights = torch.softmax(torch.randn(T, TOP_K, device="cuda"), -1).contiguous()
    summary = torch.empty(T, R, device="cuda")
    lx("moe_summary", ids, weights, emb, summary, T, R, TOP_K)
    want = (weights.double()[:, :, None] * emb[ids.long()].double()).sum(1)
    close(summary, want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("T,E,R,top_k", [(3, 5, 7, 2), (2, 9, 13, 9)])
def test_moe_summary_small_odd(lx, T, E, R, top_k):
    torch.manual_seed(T * 100 + E)
    emb = torch.randn(E, R, device="cuda").to(torch.bfloat16).contiguous()
    ids = torch.rand(T, E, device="cuda").argsort(-1)[:, :top_k].to(torch.int32).contiguous()
    weights = torch.softmax(torch.randn(T, top_k, device="cuda"), -1).contiguous()
    summary = torch.empty(T, R, device="cuda")
    lx("moe_summary", ids, weights, emb, summary, T, R, top_k)
    want = (weights.double()[:, :, None] * emb[ids.long()].double()).sum(1)
    close(summary, want, atol=1e-5, rtol=1e-5)
