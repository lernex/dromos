"""Compare two engine runs of the same prompts, logit for logit.

    python tools/compare_runs.py PROMPTS_JSON RUN_A RUN_B

For each prompt: the mean top-20 KL (A || B, as tools/compare.py computes it against the native
reference), the largest absolute logit difference, how many positions are bit-identical, and how
often the top tokens agree. Used to check that an offloaded run matches the fully resident one.
"""

import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from compare import topk_kl  # noqa: E402


def main():
    prompts = json.loads(pathlib.Path(sys.argv[1]).read_text())
    a_dir, b_dir = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    for name, ids in prompts.items():
        read = lambda d: torch.frombuffer(bytearray((d / f"{name}.f32").read_bytes()), dtype=torch.float32).view(len(ids), -1)
        a, b = read(a_dir), read(b_dir)
        kl = topk_kl(b, a)
        same = (a == b).all(-1).float().sum().item()
        agree = (a.argmax(-1) == b.argmax(-1)).float().mean()
        print(f"{name:20s} tokens {len(ids):4d}  KL {kl.mean():.3g}  max|diff| {(a - b).abs().max():.3g}  "
              f"identical rows {int(same)}/{len(ids)}  top agree {agree:.1%}")


if __name__ == "__main__":
    main()
