"""The block drafter's training code against plain references (CPU; no engine needed).

    $PY -m pytest -q tests/draft
"""

import math
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools" / "draft"))
import block_drafter as bd  # noqa: E402
import train_block as tb  # noqa: E402


def small_config(**over):
    cfg = {
        "format": bd.FORMAT, "hidden_size": 64, "vocab_size": 97, "taps": [[0, 1], [1, -1]],
        "num_layers": 2, "num_heads": 4, "num_kv_heads": 2, "head_dim": 16, "intermediate_size": 96,
        "block_size": 4, "window": 5, "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
    }
    cfg.update(over)
    return cfg


def randomize(model, gen):
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * (0.3 if p.ndim > 1 else 1.0) + (1.0 if p.ndim == 1 else 0.0))


def reference_block(model, embedding, taps, tokens, a):
    """One block at anchor a, written out position by position and head by head."""
    cfg = model.cfg
    B, W, H, KVH, Dh, I = (cfg[k] for k in ("block_size", "window", "num_heads", "num_kv_heads", "head_dim",
                                           "intermediate_size"))
    eps, theta = cfg["rms_norm_eps"], cfg["rope_theta"]

    def norm(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    def rope(x, pos):  # x [heads, Dh]
        half = Dh // 2
        out = x.clone()
        for i in range(half):
            ang = pos * theta ** (-2.0 * i / Dh)
            c, s = math.cos(ang), math.sin(ang)
            out[:, i] = x[:, i] * c - x[:, i + half] * s
            out[:, i + half] = x[:, i + half] * c + x[:, i] * s
        return out

    ctx = list(range(max(0, a - W), a))
    fused = norm(taps[ctx] @ model.fc.weight.t(), model.fc_norm) if ctx else None
    x = torch.stack([embedding[tokens[a]]] + [model.mask] * (B - 1))
    for layer in model.layers:
        h = norm(x, layer.in_norm)
        q = [rope(norm((h[j] @ layer.q.weight.t()).view(H, Dh), layer.q_norm), a + j) for j in range(B)]
        keys, values = [], []
        for i, t in enumerate(ctx):
            keys.append(rope(norm((fused[i] @ layer.k.weight.t()).view(KVH, Dh), layer.k_norm), t))
            values.append((fused[i] @ layer.v.weight.t()).view(KVH, Dh))
        for j in range(B):
            keys.append(rope(norm((h[j] @ layer.k.weight.t()).view(KVH, Dh), layer.k_norm), a + j))
            values.append((h[j] @ layer.v.weight.t()).view(KVH, Dh))
        K, Vv = torch.stack(keys), torch.stack(values)  # [n, KVH, Dh]
        att = torch.zeros(B, H, Dh, dtype=x.dtype)
        for j in range(B):
            for hd in range(H):
                kv = hd // (H // KVH)
                w = torch.softmax(K[:, kv] @ q[j][hd] / math.sqrt(Dh), 0)
                att[j, hd] = w @ Vv[:, kv]
        x = x + att.view(B, H * Dh) @ layer.o.weight.t()
        gu = norm(x, layer.post_norm) @ layer.gate_up.weight.t()
        x = x + (torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]) @ layer.down.weight.t()
    out = norm(x[1:], model.norm)
    return out, out @ model.conf.weight.t() + model.conf.bias


def test_block_forward_matches_reference():
    gen = torch.Generator().manual_seed(0)
    cfg = small_config()
    model = bd.BlockDrafter(cfg).double()
    randomize(model, gen)
    T, D = 13, cfg["hidden_size"]
    embedding = torch.randn(cfg["vocab_size"], D, generator=gen, dtype=torch.float64)
    taps = torch.randn(T, len(cfg["taps"]) * D, generator=gen, dtype=torch.float64)
    tokens = torch.randint(0, cfg["vocab_size"], (T,), generator=gen)
    anchors = torch.tensor([1, 2, 6, 9, 12])  # windows shorter than W, full, and blocks past the end
    out, conf = model(embedding, taps, tokens, anchors)
    for i, a in enumerate(anchors.tolist()):
        ref_out, ref_conf = reference_block(model, embedding, taps, tokens, a)
        torch.testing.assert_close(out[i], ref_out, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(conf[i], ref_conf, rtol=1e-9, atol=1e-9)


def test_draft_terms_gradient():
    gen = torch.Generator().manual_seed(1)
    r, D, V, U = 37, 24, 301, 11
    h = torch.randn(r, D, generator=gen, requires_grad=True)
    embedding = torch.randn(V, D, generator=gen)
    p_u = torch.softmax(torch.randn(U, V, generator=gen) * 3, -1)
    inv = torch.randint(0, U, (r,), generator=gen)
    tok = torch.randint(0, V, (r,), generator=gen)
    w_ce, w_tv = torch.rand(r, generator=gen), torch.rand(r, generator=gen)
    t_arg = p_u.argmax(-1)

    ce, tv, alpha, greedy = tb.DraftTerms.apply(h, embedding, p_u, t_arg, inv, tok)
    (w_ce * ce + w_tv * tv).sum().backward()
    g_fused = h.grad.clone()

    h.grad = None
    logq = (h @ embedding.t()).log_softmax(-1)
    q = logq.exp()
    ce_ref = -logq.gather(-1, tok[:, None])[:, 0]
    tv_ref = 0.5 * (q - p_u[inv]).abs().sum(-1)
    (w_ce * ce_ref + w_tv * tv_ref).sum().backward()
    torch.testing.assert_close(ce, ce_ref.detach())
    torch.testing.assert_close(tv, tv_ref.detach())
    torch.testing.assert_close(alpha, torch.minimum(p_u[inv], q).sum(-1).detach())
    torch.testing.assert_close(greedy, (q.argmax(-1) == t_arg[inv]).float())
    torch.testing.assert_close(g_fused, h.grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("family,taps,block", [
    ("praxis", [[0, 1], [0, 6], [0, 11], [1, 4], [1, 9], [1, -1]], 7),
    ("logos", [[0, 1], [0, 10], [0, 19], [1, 8], [1, 17], [1, -1]], 6),
])
def test_default_config(family, taps, block):
    shapes = {"praxis": (2048, 12, 8), "logos": (2560, 20, 10)}[family]
    D, L, kvh = shapes
    cfg = bd.default_config({"hidden_size": D, "num_hidden_layers": L, "num_passes": 2, "num_key_value_heads": kvh,
                             "vocab_size": 65536, "rms_norm_eps": 1e-6, "family": family})
    assert cfg["taps"] == taps
    assert cfg["block_size"] == block
    assert cfg["num_heads"] * cfg["head_dim"] == D


def test_pick_anchors_only_where_a_response_follows():
    mask = torch.tensor([0, 0, 0, 1, 1, 1, 0, 1], dtype=torch.bool)
    anchors = tb.pick_anchors(mask, 4, 100, torch.Generator().manual_seed(0))
    assert anchors.tolist() == [2, 3, 4, 6]  # a + 1 in the response, a >= 1, a + 1 < T
