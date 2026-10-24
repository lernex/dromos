"""Preflight for draft training: the shards' final-state tap must rebuild the target's logits.

    build/lernex eval MODEL_DIR PROMPTS_JSON LOGITS_DIR
    $PY tools/draft/check_features.py MODEL_DIR PROMPTS_JSON LOGITS_DIR SHARD_DIR

PROMPTS_JSON maps names to token lists; tools/draft/check_features.py jsonl PROMPTS_JSON OUT.jsonl
writes them as the features command's input. The trainer takes the target's next-token distribution
at every position as softmax(embedding @ rmsnorm(final state)); this compares that, from the bf16
shard rows, against the engine's own logits for the same tokens, and prints the worst differences.
"""

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import block_drafter as bd  # noqa: E402
from train_block import read_shard  # noqa: E402


def write_jsonl(prompts_path, out_path):
    prompts = json.loads(pathlib.Path(prompts_path).read_text())
    with open(out_path, "w") as f:
        for ids in prompts.values():
            f.write(json.dumps({"tokens": ids, "mask": [1] * len(ids)}) + "\n")


def main():
    if sys.argv[1] == "jsonl":
        write_jsonl(sys.argv[2], sys.argv[3])
        return
    model_dir, prompts_path, logits_dir, shard_dir = sys.argv[1:5]
    dev = torch.device("cuda")
    cfg = json.loads((pathlib.Path(model_dir) / "config.json").read_text())
    D, V = cfg["hidden_size"], cfg["vocab_size"]
    emb, norm = bd.load_target_head(model_dir, dev)
    prompts = json.loads(pathlib.Path(prompts_path).read_text())
    by_tokens = {tuple(ids): name for name, ids in prompts.items()}
    worst_kl, worst_top1 = 0.0, 1.0
    for shard in sorted(pathlib.Path(shard_dir).glob("shard-*.bin")):
        for tokens, _, feats in read_shard(shard):
            name = by_tokens[tuple(tokens.tolist())]
            T = len(tokens)
            ref = np.fromfile(pathlib.Path(logits_dir) / f"{name}.f32", dtype=np.float32).reshape(-1, V)[:T]
            ref = torch.from_numpy(ref).to(dev)
            final = torch.from_numpy(feats[:, -D:].copy()).to(dev).view(torch.bfloat16)
            ours = (bd.target_head_input(final, norm, cfg["rms_norm_eps"]) @ emb.t()).float()
            lp, lq = ref.log_softmax(-1), ours.log_softmax(-1)
            kl = (lp.exp() * (lp - lq)).sum(-1)
            top1 = (ref.argmax(-1) == ours.argmax(-1)).float().mean().item()
            print(f"{name:12s} T={T:5d} KL(engine || rebuilt) mean {kl.mean().item():.2e} max {kl.max().item():.2e} "
                  f"top-1 agreement {top1:.4f}")
            worst_kl, worst_top1 = max(worst_kl, kl.max().item()), min(worst_top1, top1)
    print(f"worst KL {worst_kl:.2e}, worst top-1 agreement {worst_top1:.4f}")


if __name__ == "__main__":
    main()
