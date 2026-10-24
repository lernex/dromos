"""The Lernex block drafter (format lernex.block/1): a DFlash/DSpark-style draft model for Metis.

One forward pass drafts a whole block. Slot 0 holds the pending token (the anchor, emitted but not
yet run through the target), slots 1 .. B-1 a learned mask embedding. Every slot attends to every
slot of its block (bidirectional) and to the target's fused hidden features over the last `window`
committed positions, which are injected as extra keys and values in every layer (DFlash's KV
injection). Slot j predicts the token at anchor + j through the target's frozen, tied LM head, so
the drafter owns only its layers, the fusion and two small heads:

    fused[t]  = rmsnorm(fc(concat of the target's taps at t))              t < anchor, last `window`
    x[0]      = embedding[anchor],  x[1 .. B-1] = mask
    per layer:  q, k, v from rmsnorm(x); keys and values also from fused (k_norm, RoPE on q and k)
                x += o(attention over [fused window, block]);  x += down(silu(gate) * up)
    logits[j] = embedding @ rmsnorm(x[j])                                   j = 1 .. B-1
    confidence[j] = sigmoid(conf(rmsnorm(x[j])))   [greedy, sampled]: the chance draft j holds once
                                                   the drafts before it held (DSpark's head)

The engine's BlockDrafter (engine/runtime/block_drafter.cpp) computes exactly this from the same
files: config.json and model.safetensors (bf16). Positions are absolute; RoPE angles are taken in
float64 on both sides, so long contexts lose nothing to float32 rounding of large angles.
"""

import json
import math
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

FORMAT = "lernex.block/1"


def default_config(target_config):
    """The drafter for a Metis target: five taps spread over the virtual depth (pass * layers +
    layer, from the second to the third-to-last, as DFlash spreads its five) plus the final state,
    three layers at the target's width, and a block as long as one engine verify step."""
    D = target_config["hidden_size"]
    L, P = target_config["num_hidden_layers"], target_config["num_passes"]
    depth = L * P
    spread = sorted({round(1 + i * (depth - 4) / 4) for i in range(5)})
    taps = [[v // L, v % L] for v in spread] + [[P - 1, -1]]
    kvh = target_config["num_key_value_heads"]
    return {
        "format": FORMAT,
        "target_family": target_config.get("family", ""),
        "hidden_size": D,
        "vocab_size": target_config["vocab_size"],
        "taps": taps,
        "num_layers": 3,
        "num_heads": D // 64,
        "num_kv_heads": kvh,
        "head_dim": 64,
        "intermediate_size": 2 * D,
        # the anchor and block_size - 1 drafts make one verify step (MetisModel::max_verify_tokens)
        "block_size": min(8, 63 // kvh),
        "window": 256,
        "rope_theta": 10000.0,
        "rms_norm_eps": target_config["rms_norm_eps"],
    }


def wide(dtype):
    """The dtype to compute in: float32, or float64 for float64 inputs (the tests)."""
    return torch.promote_types(dtype, torch.float32)


def rms_norm(x, weight, eps):
    xf = x.to(wide(x.dtype))
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.to(xf.dtype)).to(x.dtype)


def rope_tables(positions, head_dim, theta):
    """cos, sin [..., head_dim / 2] (float64) for absolute positions, angles in float64."""
    inv = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float64, device=positions.device) / head_dim)
    angle = positions.to(torch.float64)[..., None] * inv
    return torch.cos(angle), torch.sin(angle)


def apply_rope(x, cos, sin):
    """Rotate halves (x[i], x[i + Dh/2]), the Llama layout. x [..., heads, Dh]; cos, sin [..., Dh/2]."""
    h = x.shape[-1] // 2
    cdt = wide(x.dtype)
    x1, x2 = x[..., :h].to(cdt), x[..., h:].to(cdt)
    c, s = cos.unsqueeze(-2).to(cdt), sin.unsqueeze(-2).to(cdt)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)


class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D, H, KVH, Dh, I = (cfg[k] for k in ("hidden_size", "num_heads", "num_kv_heads", "head_dim", "intermediate_size"))
        self.H, self.KVH, self.Dh, self.I, self.eps = H, KVH, Dh, I, cfg["rms_norm_eps"]
        self.in_norm = nn.Parameter(torch.ones(D))
        self.q = nn.Linear(D, H * Dh, bias=False)
        self.k = nn.Linear(D, KVH * Dh, bias=False)
        self.v = nn.Linear(D, KVH * Dh, bias=False)
        self.o = nn.Linear(H * Dh, D, bias=False)
        self.q_norm = nn.Parameter(torch.ones(Dh))
        self.k_norm = nn.Parameter(torch.ones(Dh))
        self.post_norm = nn.Parameter(torch.ones(D))
        self.gate_up = nn.Linear(D, 2 * I, bias=False)
        self.down = nn.Linear(I, D, bias=False)

    def context_kv(self, fused, cos, sin):
        """Keys and values of the fused target features [T, D] at their positions."""
        T = fused.shape[0]
        k = rms_norm(self.k(fused).view(T, self.KVH, self.Dh), self.k_norm, self.eps)
        return apply_rope(k, cos, sin), self.v(fused).view(T, self.KVH, self.Dh)

    def forward(self, x, cos, sin, k_ctx, v_ctx, ctx_valid):
        """x [A, B, D] blocks at positions given by cos, sin [A, B, Dh/2]; k_ctx, v_ctx [A, W, KVH, Dh]
        each block's context window, ctx_valid [A, W]."""
        A, B, _ = x.shape
        h = rms_norm(x, self.in_norm, self.eps)
        q = apply_rope(rms_norm(self.q(h).view(A, B, self.H, self.Dh), self.q_norm, self.eps), cos, sin)
        k = apply_rope(rms_norm(self.k(h).view(A, B, self.KVH, self.Dh), self.k_norm, self.eps), cos, sin)
        v = self.v(h).view(A, B, self.KVH, self.Dh)
        keys = torch.cat([k_ctx, k], dim=1).transpose(1, 2)  # [A, KVH, W + B, Dh]
        values = torch.cat([v_ctx, v], dim=1).transpose(1, 2)
        mask = torch.cat([ctx_valid, ctx_valid.new_ones(A, B)], dim=1)[:, None, None, :]
        # GQA as batched products: query head h reads kv head h // G, so a kv head's G * B query
        # rows go in one product (with 7 queries per block this beats the fused kernels' backward)
        G = self.H // self.KVH
        qg = q.view(A, B, self.KVH, G, self.Dh).permute(0, 2, 3, 1, 4).reshape(A, self.KVH, G * B, self.Dh)
        scores = (qg @ keys.transpose(-1, -2)).to(wide(qg.dtype)) * (1.0 / math.sqrt(self.Dh))
        probs = scores.masked_fill(~mask, float("-inf")).softmax(-1).to(values.dtype)
        att = (probs @ values).view(A, self.KVH, G, B, self.Dh).permute(0, 3, 1, 2, 4)
        x = x + self.o(att.reshape(A, B, self.H * self.Dh))
        gu = self.gate_up(rms_norm(x, self.post_norm, self.eps))
        return x + self.down(F.silu(gu[..., : self.I]) * gu[..., self.I :])


class BlockDrafter(nn.Module):
    """The trainable drafter. The target's embedding (its tied LM head) is passed in, not owned."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D = cfg["hidden_size"]
        self.eps = cfg["rms_norm_eps"]
        self.fc = nn.Linear(len(cfg["taps"]) * D, D, bias=False)
        self.fc_norm = nn.Parameter(torch.ones(D))
        self.mask = nn.Parameter(torch.randn(D) * 0.02)
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg["num_layers"]))
        self.norm = nn.Parameter(torch.ones(D))
        self.conf = nn.Linear(D, 2)
        nn.init.zeros_(self.conf.weight)
        nn.init.zeros_(self.conf.bias)

    def fuse(self, taps):
        return rms_norm(self.fc(taps), self.fc_norm, self.eps)

    def forward(self, embedding, taps, tokens, anchors):
        """Blocks at `anchors` of one sequence. taps [T, n_taps * D] (the drafter's inputs), tokens
        [T], anchors [A] (each >= 1, the position of its slot 0). Returns the normalized outputs of
        slots 1 .. B-1, [A, B - 1, D] (logits are embedding @ these) and the confidence logits
        [A, B - 1, 2]."""
        cfg = self.cfg
        B, W, Dh = cfg["block_size"], cfg["window"], cfg["head_dim"]
        T, A = tokens.shape[0], anchors.shape[0]
        dev = tokens.device
        fused = self.fuse(taps)
        ctx_cos, ctx_sin = rope_tables(torch.arange(T, device=dev), Dh, cfg["rope_theta"])
        idx = anchors[:, None] - W + torch.arange(W, device=dev)[None, :]  # [A, W]
        valid = idx >= 0
        idx = idx.clamp(min=0)
        blk_pos = anchors[:, None] + torch.arange(B, device=dev)[None, :]
        cos, sin = rope_tables(blk_pos, Dh, cfg["rope_theta"])
        x = torch.cat([embedding[tokens[anchors]][:, None, :].to(fused.dtype),
                       self.mask.to(fused.dtype).expand(A, B - 1, -1)], dim=1)
        flat = idx.reshape(-1)
        for layer in self.layers:
            kc, vc = layer.context_kv(fused, ctx_cos, ctx_sin)
            # index_select: its backward is an index_add, far faster than fancy indexing's index_put
            kw = kc.index_select(0, flat).view(A, W, *kc.shape[1:])
            vw = vc.index_select(0, flat).view(A, W, *vc.shape[1:])
            x = layer(x, cos, sin, kw, vw, valid)
        out = rms_norm(x[:, 1:], self.norm, self.eps)
        return out, self.conf(out.detach().to(wide(out.dtype)))


def target_head_input(final_state, final_norm, eps):
    """The target's LM head input from its final state (the tap (passes - 1, -1))."""
    return rms_norm(final_state, final_norm, eps)


def load_target_head(target_dir, device):
    """The target's tied embedding [V, D] and final norm [D] (bf16), straight from its checkpoint."""
    from safetensors import safe_open

    out = {}
    for f in sorted(pathlib.Path(target_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device=str(device)) as st:
            for name in ("embedding.weight", "final_norm.weight"):
                if name in st.keys():
                    out[name] = st.get_tensor(name)
    if len(out) != 2:
        raise RuntimeError(f"embedding.weight / final_norm.weight not found in {target_dir}")
    return out["embedding.weight"], out["final_norm.weight"]


def save(model, out_dir, extra=None):
    """config.json + model.safetensors (bf16), the format the engine loads."""
    from safetensors.torch import save_file

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    save_file(tensors, str(out / "model.safetensors.tmp"))
    (out / "model.safetensors.tmp").rename(out / "model.safetensors")
    cfg = dict(model.cfg)
    if extra:
        cfg.update(extra)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))


def load(draft_dir, device, dtype=torch.float32):
    from safetensors.torch import load_file

    cfg = json.loads((pathlib.Path(draft_dir) / "config.json").read_text())
    if cfg.get("format") != FORMAT:
        raise RuntimeError(f"{draft_dir}: not a {FORMAT} drafter")
    model = BlockDrafter(cfg)
    model.load_state_dict({k: v.to(dtype) for k, v in load_file(str(pathlib.Path(draft_dir) / "model.safetensors")).items()})
    return model.to(device=device, dtype=dtype)
