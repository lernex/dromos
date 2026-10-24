"""Compare engine logits with the FP32 reference logits of the native runtime.

    python tools/compare.py REFS_DIR ENGINE_DIR

REFS_DIR holds prompts.json ({name: token ids}) and native_logits.pt ({name: [T, V]}).
ENGINE_DIR holds NAME.f32 for each prompt: raw little-endian float32 logits, [T, V].
Reports, per prompt, the mean top-20 KL (reference || engine, both renormalized on the
engine's top 20, the metric every Metis port reports) and how often the top token agrees.
"""

import json
import pathlib
import sys

import torch


def topk_kl(engine_logits, ref_logits, k=20):
    lp = torch.log_softmax(engine_logits.float(), -1)
    lr = torch.log_softmax(ref_logits.float(), -1)
    top = lp.topk(k, dim=-1).indices
    v = lp.gather(-1, top)
    r = lr.gather(-1, top)
    v = v - torch.logsumexp(v, -1, keepdim=True)
    r = r - torch.logsumexp(r, -1, keepdim=True)
    return (r.exp() * (r - v)).sum(-1)


def main():
    refs, engine = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    prompts = json.loads((refs / "prompts.json").read_text())
    native = torch.load(refs / "native_logits.pt")
    for name, ids in prompts.items():
        path = engine / f"{name}.f32"
        if not path.exists():
            print(f"{name}: missing {path}")
            continue
        ref = native[name].float()
        got = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.float32).view(len(ids), -1)
        kl = topk_kl(got, ref)
        agree = (got.argmax(-1) == ref.argmax(-1)).float().mean()
        print(f"{name:20s} tokens {len(ids):4d}  KL {kl.mean():.4g}  max {kl.max():.3g}  top agree {agree:.1%}")


if __name__ == "__main__":
    main()
