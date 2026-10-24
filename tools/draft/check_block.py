"""The engine's block drafter against the PyTorch model it was trained as, on the same inputs.

    build/lernex block-check MODEL_DIR DRAFT_DIR PROMPTS_JSON OUT_DIR
    $PY tools/draft/check_block.py MODEL_DIR DRAFT_DIR OUT_DIR

    $PY tools/draft/check_block.py init MODEL_DIR DRAFT_DIR   # a freshly initialized drafter to check

A fresh drafter is the stronger check: every input path (context window, anchor, mask) moves its
outputs, while a briefly trained one can ignore most of them.

block-check saves, per prompt, the target's feature rows over the drafter's window, the anchor and
the engine's logits and confidences for one block. This recomputes the block with
block_drafter.BlockDrafter on the same bf16 weights, with every product's input rounded to bf16 as
the engine does (and as bf16 autocast did in training), twice: in float64, the reference, and in
float32, which sums in a different order and so lands some of those roundings on the other side.
The float32 run's distance from the reference is the noise floor of bf16 activations; the engine
must agree on every row's argmax and stay within FLOOR_FACTOR times that floor (logits relative to
the row's largest |logit|, and confidences). A wrong layer, position or mask is off by orders of
magnitude more.
"""

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import block_drafter as bd  # noqa: E402

FLOOR_FACTOR = 4.0
FLOOR_MIN = 1e-5  # floors below this (no rounding landed differently) count as this

def init(model_dir, draft_dir):
    torch.manual_seed(0)
    cfg = bd.default_config(json.loads((pathlib.Path(model_dir) / "config.json").read_text()))
    model = bd.BlockDrafter(cfg)
    with torch.no_grad():  # norms and the confidence head away from their neutral starting values
        for name, p in model.named_parameters():
            if p.ndim == 1 or name.startswith("conf"):
                p.add_(torch.randn_like(p) * 0.2)
    bd.save(model, draft_dir)


def main():
    if sys.argv[1] == "init":
        init(sys.argv[2], sys.argv[3])
        return
    model_dir, draft_dir, out_dir = sys.argv[1:4]
    dev = torch.device("cuda")
    models = {}
    for dtype in (torch.float64, torch.float32):
        model = bd.load(draft_dir, dev, dtype).eval()
        for mod in model.modules():  # the engine's products read bf16 activations
            if isinstance(mod, torch.nn.Linear):
                mod.register_forward_pre_hook(lambda _, args: (args[0].bfloat16().to(args[0].dtype),))
        models[dtype] = model
    emb, _ = bd.load_target_head(model_dir, dev)
    V = emb.shape[0]

    def block(dtype, taps, tokens, P):
        with torch.no_grad():
            out, conf = models[dtype](emb.to(dtype), taps.to(dtype), tokens, torch.tensor([P], device=dev))
            return out[0].bfloat16().to(dtype) @ emb.to(dtype).t(), torch.sigmoid(conf[0]).flatten().cpu()

    def rel(a, b):
        a, b = a.double(), b.double()
        return ((a - b).abs().max(-1).values / b.abs().max(-1).values).max().item()

    ok = True
    for meta_path in sorted(pathlib.Path(out_dir).glob("*.json")):
        name = meta_path.stem
        meta = json.loads(meta_path.read_text())
        P, first, F, rows = meta["position"], meta["first"], meta["feature_dim"], meta["rows"]
        feats = np.fromfile(pathlib.Path(out_dir) / f"{name}.features.f32", dtype=np.float32).reshape(-1, F)
        engine = torch.from_numpy(np.fromfile(pathlib.Path(out_dir) / f"{name}.logits.f32", dtype=np.float32)
                                  .reshape(rows, V)).to(dev)
        engine_conf = torch.tensor(meta["confidence"], dtype=torch.float64)
        taps = torch.zeros(P + 1, F, device=dev)
        taps[first:P] = torch.from_numpy(feats).to(dev)
        tokens = torch.zeros(P + 1, dtype=torch.long, device=dev)
        tokens[P] = meta["anchor"]
        ref, ref_conf = block(torch.float64, taps, tokens, P)
        f32, f32_conf = block(torch.float32, taps, tokens, P)
        d_engine, d_floor = rel(engine, ref), max(rel(f32, ref), FLOOR_MIN)
        c_engine = (engine_conf - ref_conf).abs().max().item()
        c_floor = max((f32_conf.double() - ref_conf).abs().max().item(), FLOOR_MIN)
        same = bool((engine.argmax(-1) == ref.argmax(-1)).all())
        good = same and d_engine <= FLOOR_FACTOR * d_floor and c_engine <= FLOOR_FACTOR * c_floor
        ok &= good
        print(f"{name:14s} anchor at {P:5d}: logits {d_engine:.1e} from the reference (float32 {d_floor:.1e}), "
              f"confidence {c_engine:.1e} (float32 {c_floor:.1e}), argmax {'agrees' if same else 'DIFFERS'} on all "
              f"{rows} rows{'' if good else '  FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
