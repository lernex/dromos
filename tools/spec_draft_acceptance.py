"""How often a draft model's proposals would hold against a target model, measured on real outputs.

    # 1. the target's continuations, with its top-20 log-probabilities at every step (vLLM, any format)
    ~/venvs/vllm-metis/bin/python tools/spec_draft_acceptance.py target TARGET_DIR PROMPTS_JSON OUT_DIR
    # 2. the drafter's logits over prompt + continuation (the engine), then the analysis
    $PY tools/spec_draft_acceptance.py drafter OUT_DIR
    build/lernex eval DRAFT_DIR OUT_DIR/teacher.json OUT_DIR/logits
    $PY tools/spec_draft_acceptance.py analyze OUT_DIR [COSTS]

Greedy: a draft holds while it equals the target's token, so along the target's own greedy output
the drafter's proposals are its argmax given the target's prefix, until the first miss. The run of
matches from each position gives the accepted count of a k-draft step exactly, and walking the
sequence step by step gives the mean tokens per step for each k.

Sampled (temperature, top-p, top-k = 20): the target's filtered distribution p at a step is fixed by
its top-20 raw log-probabilities (top-k 20 keeps exactly those). A draft drawn from the drafter's q
holds with probability sum_x min(p(x), q(x)); the analysis reports that mean along the target's
sampled output, and the tokens per step it implies for each k if positions held independently.
"""

import json
import math
import pathlib
import sys

GREEDY_TOKENS = 512
FILTER = dict(temperature=0.6, top_p=0.95, top_k=20)


def target(target_dir, prompts_path, out_dir):
    from vllm import LLM, SamplingParams

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(pathlib.Path(prompts_path).read_text())
    llm = LLM(model=target_dir, trust_remote_code=True, max_model_len=4096, gpu_memory_utilization=0.85,
              enforce_eager=True, max_num_seqs=16, seed=0)
    names = list(prompts)
    inputs = [{"prompt_token_ids": prompts[n]} for n in names]
    greedy = SamplingParams(temperature=0.0, max_tokens=GREEDY_TOKENS, logprobs=20)
    sampled = SamplingParams(max_tokens=GREEDY_TOKENS, logprobs=20, seed=1, **FILTER)
    result = {}
    for mode, params in (("greedy", greedy), ("sampled", sampled)):
        outs = llm.generate(inputs, params)
        for n, o in zip(names, outs):
            c = o.outputs[0]
            top = [[(int(t), float(lp.logprob)) for t, lp in step.items()] for step in c.logprobs]
            result.setdefault(n, {})[mode] = {"tokens": list(c.token_ids), "top": top}
    result["_prompts"] = prompts
    (out / "target.json").write_text(json.dumps(result))
    print("target continuations:", {n: len(result[n]["greedy"]["tokens"]) for n in names})


def drafter(out_dir):
    out = pathlib.Path(out_dir)
    data = json.loads((out / "target.json").read_text())
    teacher = {}
    for n, ids in data["_prompts"].items():
        for mode in ("greedy", "sampled"):
            teacher[f"{n}.{mode}"] = ids + data[n][mode]["tokens"]
    (out / "teacher.json").write_text(json.dumps(teacher))
    print(f"wrote {out / 'teacher.json'}: run `lernex eval DRAFT_DIR {out / 'teacher.json'} {out / 'logits'}`")


def filtered(logits, temperature, top_p, top_k):
    """lx_spec_probs's distribution over a dict {token: logit} (the candidates)."""
    items = sorted(logits.items(), key=lambda kv: -kv[1])[:top_k]
    scaled = [(t, l / temperature) for t, l in items]
    top = scaled[0][1]
    e = [(t, math.exp(s - top)) for t, s in scaled]
    total = sum(w for _, w in e)
    kept, acc = [], 0.0
    for t, w in e:
        kept.append((t, w))
        acc += w
        if acc >= top_p * total:
            break
    z = sum(w for _, w in kept)
    return {t: w / z for t, w in kept}


def analyze(out_dir, costs):
    import numpy as np
    import torch

    out = pathlib.Path(out_dir)
    data = json.loads((out / "target.json").read_text())
    V = 65536
    report = {}
    all_runs, all_alpha = [], []
    for n, ids in data["_prompts"].items():
        P = len(ids)
        # greedy: does the drafter's argmax after the target's prefix equal the target's next token
        toks = data[n]["greedy"]["tokens"]
        logits = np.memmap(out / "logits" / f"{n}.greedy.f32", dtype=np.float32, mode="r").reshape(-1, V)
        rows = torch.from_numpy(np.array(logits[P - 1 : P - 1 + len(toks)]))
        match = (rows.argmax(-1) == torch.tensor(toks)).tolist()
        all_runs.append(match)
        # sampled: sum_x min(p, q) at each step
        st = data[n]["sampled"]
        logits_s = np.memmap(out / "logits" / f"{n}.sampled.f32", dtype=np.float32, mode="r").reshape(-1, V)
        alphas = []
        for i, top in enumerate(st["top"]):
            p = filtered(dict(top), **FILTER)
            ql = torch.from_numpy(np.array(logits_s[P - 1 + i]))
            # q: the drafter's distribution under the same filter, over its whole vocabulary
            qv, qi = torch.topk(ql, FILTER["top_k"])
            q = filtered({int(t): float(v) for t, v in zip(qi, qv)}, **FILTER)
            alphas.append(sum(min(p.get(t, 0.0), q.get(t, 0.0)) for t in set(p) | set(q)))
        all_alpha.append(alphas)
        report[n] = {"greedy_match": sum(match) / len(match), "sampled_alpha": sum(alphas) / len(alphas)}

    def walk(match, k):
        """Tokens per step of greedy speculative decoding with k drafts along a match sequence. match[i]
        says the drafter's argmax predicts token i from the target's prefix before it."""
        i, steps, emitted = 0, 0, 0
        while i < len(match):
            a = 0
            while a < k and i + a + 1 < len(match) and match[i + a + 1]:
                a += 1
            steps += 1
            emitted += a + 1
            i += a + 1
        return emitted / steps

    def expected(alphas, k):
        """Tokens per step if position j's draft held independently with probability alpha_j."""
        i, steps, emitted = 0, 0, 0.0
        while i < len(alphas) - 1:
            reach, e = 1.0, 1.0
            for j in range(1, k + 1):
                if i + j >= len(alphas):
                    break
                reach *= alphas[i + j]
                e += reach
            steps += 1
            emitted += e
            i += max(1, int(round(e)))
        return emitted / max(steps, 1)

    print(f"{'prompt':12s} greedy-match  sampled-alpha   greedy tokens/step k=1..6          sampled tokens/step k=1..6")
    for (n, r), match, alphas in zip(report.items(), all_runs, all_alpha):
        g = [walk(match, k) for k in range(1, 7)]
        s = [expected(alphas, k) for k in range(1, 7)]
        r["greedy_tokens_per_step"] = g
        r["sampled_tokens_per_step"] = s
        print(f"{n:12s} {r['greedy_match']:11.1%} {r['sampled_alpha']:13.1%}   " + " ".join(f"{v:4.2f}" for v in g)
              + "   " + " ".join(f"{v:4.2f}" for v in s))
    (out / "acceptance.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "target":
        target(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "drafter":
        drafter(sys.argv[2])
    elif cmd == "analyze":
        analyze(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
