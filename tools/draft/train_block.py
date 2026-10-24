"""Train a lernex.block/1 drafter on the target's hidden states, streamed from `lernex features`.

    $PY tools/draft/train_block.py TARGET_DIR SHARD_DIR OUT_DIR --total-seqs N [options]

SHARD_DIR is the queue `lernex features` writes (cli/features.cpp): shards appear whole, the trainer
reads them in name order and deletes each one it has read (--keep leaves them), and DONE ends the
stream. Each shard's features are the drafter's taps (config["taps"], the last one the target's
final state), so the target's own next-token distribution at every position is
softmax(embedding @ rmsnorm(final state)): the trainer rebuilds it exactly, with no logits stored.

Loss per anchor block (DSpark's recipe, position weights exp(-(j - 1) / gamma)):
    0.1 * CE(draft, the target's token) + 0.9 * TV(draft, target) + BCE(confidence, holds)
where holds is [argmax draft == argmax target, sum min(p, q)] (greedy and sampled acceptance).
Up to --anchors random anchors per sequence, on response tokens (mask = 1). Prints, as it goes, the
tokens per verify step a chain of drafts would give: greedy from the run of matching argmaxes,
sampled from the products of sum min(p, q).

--eval DIR (a second, kept shard dir) is scored every --eval-every steps. OUT_DIR gets the drafter
(config.json, model.safetensors: what the engine loads) every --save-every steps and at the end,
and state.pt (weights, optimizer, step, data position) to resume with --resume.
"""

import argparse
import json
import math
import os
import pathlib
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import block_drafter as bd  # noqa: E402

CE_W, TV_W, CONF_W = 0.1, 0.9, 1.0


def read_shard(path):
    """[(tokens int32 [T], mask uint8 [T], features bf16-as-int16 [T, F])] of one shard."""
    with open(path, "rb") as f:
        head = f.read(24)
        if head[:4] != b"LXFS":
            raise RuntimeError(f"{path}: not a feature shard")
        version, n, fd, total = struct.unpack("<IIIQ", head[4:])
        if version != 1:
            raise RuntimeError(f"{path}: shard version {version}")
        lengths = np.frombuffer(f.read(4 * n), dtype=np.uint32)
        tokens = np.frombuffer(f.read(4 * total), dtype=np.int32)
        mask = np.frombuffer(f.read(total), dtype=np.uint8)
        feats = np.frombuffer(f.read(2 * total * fd), dtype=np.int16).reshape(total, fd)
    out, at = [], 0
    for T in lengths.tolist():
        out.append((tokens[at : at + T], mask[at : at + T], feats[at : at + T]))
        at += T
    return out


def shard_stream(shard_dir, keep, poll=1.0):
    """Sequences from a queue directory, each shard once, until DONE and nothing new."""
    d = pathlib.Path(shard_dir)
    taken = set()

    def fresh():
        return [s for s in sorted(d.glob("shard-*.bin")) if s.name not in taken]

    while True:
        if (d / "FAILED").exists():
            raise RuntimeError(f"the features producer for {d} failed (see its log)")
        shards = fresh()
        if not shards:
            # DONE is written after the last shard, so once it shows, one more look finds them all
            if (d / "DONE").exists() and not fresh():
                return
            if poll:
                time.sleep(poll)
            continue
        for s in shards:
            seqs = read_shard(s)
            taken.add(s.name)
            if not keep:
                s.unlink()
            yield from seqs


def to_device(seq, device):
    tokens, mask, feats = seq
    t = torch.from_numpy(tokens.astype(np.int64)).to(device)
    m = torch.from_numpy(mask.astype(np.bool_)).to(device)
    f = torch.from_numpy(feats.copy()).to(device).view(torch.bfloat16)
    return t, m, f


def pick_anchors(mask, B, n, gen):
    """Anchor positions a >= 1 whose first draft (a + 1) is a response token."""
    T = mask.shape[0]
    ok = torch.zeros_like(mask)
    ok[1 : T - 1] = mask[2:]
    cand = ok.nonzero()[:, 0]
    if cand.numel() > n:
        cand = cand[torch.randperm(cand.numel(), generator=gen, device="cpu")[:n].to(cand.device)]
    return cand.sort().values


class DraftTerms(torch.autograd.Function):
    """Per draft row: CE against the target's token, TV against the target's distribution, and the
    acceptance it would get (sum min(p, q), argmax match), from the drafter's normalized outputs h
    through the frozen head. Nothing [rows, V] is kept for the backward pass, which recomputes the
    logits and applies the closed-form gradient:
        d CE / d logit_i = q_i - [i = token]
        d TV / d logit_i = q_i (s_i - sum_j q_j s_j) / 2,   s = sign(q - p)"""

    @staticmethod
    def forward(ctx, h, embedding, p_u, t_arg, inv, tok):
        ql = (h.to(embedding.dtype) @ embedding.t()).float()
        logq = ql.log_softmax(-1)
        q = logq.exp()
        p = p_u[inv].float()
        ce = -logq.gather(-1, tok[:, None])[:, 0]
        tv = 0.5 * (q - p).abs().sum(-1)
        alpha = torch.minimum(p, q).sum(-1)
        greedy = (ql.argmax(-1) == t_arg[inv]).float()
        ctx.save_for_backward(h, embedding, p_u, inv, tok)
        ctx.mark_non_differentiable(alpha, greedy)
        return ce, tv, alpha, greedy

    @staticmethod
    def backward(ctx, g_ce, g_tv, _alpha, _greedy):
        h, embedding, p_u, inv, tok = ctx.saved_tensors
        q = (h.to(embedding.dtype) @ embedding.t()).float().softmax(-1)
        s = torch.sign(q - p_u[inv].float())
        g_ce = torch.zeros_like(q[:, 0]) if g_ce is None else g_ce
        g_tv = torch.zeros_like(q[:, 0]) if g_tv is None else g_tv
        grad = q * (g_ce[:, None] + 0.5 * g_tv[:, None] * (s - (q * s).sum(-1, keepdim=True)))
        grad.scatter_add_(-1, tok[:, None], -g_ce[:, None])
        return (grad.to(embedding.dtype) @ embedding).to(h.dtype), None, None, None, None, None


def block_loss(model, embedding, final_norm, seq, cfg, gamma, n_anchors, gen, chunk=1024):
    """The loss of one sequence (a scalar to backpropagate) and its acceptance statistics."""
    tokens, mask, feats = seq
    B, D = cfg["block_size"], cfg["hidden_size"]
    T = tokens.shape[0]
    anchors = pick_anchors(mask, B, n_anchors, gen)
    if anchors.numel() == 0:
        return None, None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, conf = model(embedding, feats, tokens, anchors)  # [A, B-1, D], [A, B-1, 2]
    A = anchors.numel()
    pos = anchors[:, None] + torch.arange(1, B, device=tokens.device)[None, :]  # the drafts' positions
    valid = pos < T
    pos_c = pos.clamp(max=T - 1)
    valid &= mask[pos_c]
    w = torch.exp(-(torch.arange(B - 1, device=tokens.device, dtype=torch.float32)) / gamma)[None, :].expand(A, -1)
    w = w * valid
    rows = valid.nonzero(as_tuple=True)
    h = out[rows]                                       # [R, D]
    tok = tokens[pos_c[rows]]
    wr = w[rows]
    # The target's distribution over each draft's token, once per position (blocks overlap, so far
    # fewer positions than rows), kept in bf16 outside the checkpointed chunks: never recomputed.
    uniq, inv = torch.unique(pos_c[rows] - 1, return_inverse=True)
    with torch.no_grad():
        p_u = torch.empty(uniq.numel(), embedding.shape[0], dtype=torch.bfloat16, device=tokens.device)
        t_arg = torch.empty(uniq.numel(), dtype=torch.long, device=tokens.device)
        for s in range(0, uniq.numel(), chunk):
            tin = bd.target_head_input(feats[uniq[s : s + chunk], -D:], final_norm, cfg["rms_norm_eps"])
            logits = (tin @ embedding.t()).float()
            t_arg[s : s + chunk] = logits.argmax(-1)
            p_u[s : s + chunk] = logits.softmax(-1).to(torch.bfloat16)

    ce, tv, alpha, greedy = [], [], [], []
    for s in range(0, h.shape[0], chunk):
        terms = DraftTerms.apply(h[s : s + chunk], embedding, p_u, t_arg, inv[s : s + chunk], tok[s : s + chunk])
        for acc, t in zip((ce, tv, alpha, greedy), terms):
            acc.append(t)
    ce, tv, alpha, greedy = (torch.cat(x) for x in (ce, tv, alpha, greedy))
    wsum = wr.sum().clamp(min=1e-6)
    draft_loss = (wr * (CE_W * ce + TV_W * tv)).sum() / wsum
    holds = torch.stack([greedy, alpha], dim=-1)
    conf_loss = F.binary_cross_entropy_with_logits(conf[rows].float(), holds, reduction="none").sum(-1).mean()
    loss = draft_loss + CONF_W * conf_loss

    # chains: tokens per verify step with all B - 1 drafts (1 + accepted), per anchor
    with torch.no_grad():
        g = torch.zeros(A, B - 1, device=tokens.device)
        a = torch.zeros(A, B - 1, device=tokens.device)
        g[rows], a[rows] = greedy, alpha
        g_run = torch.cumprod(g, dim=1).sum(1)
        a_run = torch.cumprod(a, dim=1).sum(1)
        full = valid.all(1)
        stats = {
            "ce": (ce * wr).sum().item() / wsum.item(),
            "tv": (tv * wr).sum().item() / wsum.item(),
            "conf": conf_loss.item(),
            "greedy_tau": (1 + g_run[full]).sum().item(), "sampled_tau": (1 + a_run[full]).sum().item(),
            "chains": full.sum().item(), "blocks": A,
            "greedy_by_pos": (g * valid).sum(0).tolist(), "alpha_by_pos": (a * valid).sum(0).tolist(),
            "count_by_pos": valid.sum(0).tolist(),
        }
    return loss, stats


def merge(total, s):
    for k, v in s.items():
        if isinstance(v, list):
            total[k] = [x + y for x, y in zip(total.get(k, [0.0] * len(v)), v)]
        else:
            total[k] = total.get(k, 0.0) + v


def describe(total, n):
    c = max(total.get("chains", 0), 1)
    cnt = [max(x, 1) for x in total.get("count_by_pos", [1])]
    return (f"ce {total['ce'] / n:.3f} tv {total['tv'] / n:.3f} conf {total['conf'] / n:.3f} "
            f"tau greedy {total['greedy_tau'] / c:.2f} sampled {total['sampled_tau'] / c:.2f} | greedy by slot "
            + " ".join(f"{g / k:.2f}" for g, k in zip(total["greedy_by_pos"], cnt)))


def evaluate(model, embedding, final_norm, eval_dir, cfg, gamma, device, limit):
    if not (pathlib.Path(eval_dir) / "DONE").exists():
        return "eval shards not complete (no DONE)"
    model.eval()
    total, n = {}, 0
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for seq in shard_stream(eval_dir, keep=True, poll=0):
            _, s = block_loss(model, embedding, final_norm, to_device(seq, device), cfg, gamma, 512, gen)
            if s:
                merge(total, s)
                n += 1
            if n >= limit:
                break
    model.train()
    return describe(total, max(n, 1)) if n else "no eval data"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target_dir")
    ap.add_argument("shard_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--total-seqs", type=int, required=True, help="sequences the stream will bring (all epochs)")
    ap.add_argument("--config", help="drafter config JSON (default: block_drafter.default_config)")
    ap.add_argument("--batch-seqs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=float, default=0.04)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--anchors", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=4.0)
    ap.add_argument("--eval", dest="eval_dir")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-seqs", type=int, default=96)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--keep", action="store_true", help="leave shards in place (smoke tests)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    target_cfg = json.loads((pathlib.Path(args.target_dir) / "config.json").read_text())
    cfg = json.loads(pathlib.Path(args.config).read_text()) if args.config else bd.default_config(target_cfg)
    embedding, final_norm = bd.load_target_head(args.target_dir, device)
    embedding.requires_grad_(False)
    model = bd.BlockDrafter(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    total_steps = max(1, args.total_seqs // args.batch_seqs)
    warm = max(1, int(args.warmup * total_steps))

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        t = min(1.0, (step - warm) / max(1, total_steps - warm))
        return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t)))

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    step, seen = 0, 0
    if args.resume and (out / "state.pt").exists():
        state = torch.load(out / "state.pt", map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step, seen = state["step"], state["seen"]
        print(f"resumed at step {step}, {seen} sequences seen")
    print(f"drafter: {n_params / 1e6:.1f}M parameters, block {cfg['block_size']}, taps {cfg['taps']}; "
          f"{total_steps} steps of {args.batch_seqs} sequences", flush=True)

    gen = torch.Generator().manual_seed(args.seed + step)
    total, n_seq, in_batch, tokens_done = {}, 0, 0, 0
    t0 = time.time()
    model.train()

    def checkpoint():
        bd.save(model, out / "draft", {"trained_steps": step, "trained_seqs": seen})
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "seen": seen}, out / "state.pt.tmp")
        os.replace(out / "state.pt.tmp", out / "state.pt")
        (out / "position.json").write_text(json.dumps({"step": step, "seen": seen}))

    for seq in shard_stream(args.shard_dir, args.keep):
        dseq = to_device(seq, device)
        loss, s = block_loss(model, embedding, final_norm, dseq, cfg, args.gamma, args.anchors, gen)
        seen += 1
        tokens_done += dseq[0].numel()
        if loss is None:
            continue
        (loss / args.batch_seqs).backward()
        merge(total, s)
        n_seq += 1
        in_batch += 1
        if in_batch < args.batch_seqs:
            continue
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        in_batch = 0
        step += 1
        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step}/{total_steps} lr {lr_at(step):.2e} |g| {gnorm:.2f} {describe(total, n_seq)} | "
                  f"{n_seq / dt:.2f} seq/s {total['blocks'] / dt:.0f} blocks/s {tokens_done / dt:.0f} tok/s, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            total, n_seq, tokens_done, t0 = {}, 0, 0, time.time()
        if args.eval_dir and step % args.eval_every == 0:
            print(f"eval step {step}: {evaluate(model, embedding, final_norm, args.eval_dir, cfg, args.gamma, device, args.eval_seqs)}", flush=True)
        if step % args.save_every == 0:
            checkpoint()
        if args.max_steps and step >= args.max_steps:
            break
    if n_seq:
        print(f"last: {describe(total, n_seq)} | {n_seq / (time.time() - t0):.2f} seq/s", flush=True)
    checkpoint()
    if args.eval_dir:
        print(f"eval final: {evaluate(model, embedding, final_norm, args.eval_dir, cfg, args.gamma, device, args.eval_seqs)}", flush=True)


if __name__ == "__main__":
    main()
