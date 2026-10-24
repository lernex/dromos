"""The target answers each training conversation; the answers are what the drafter learns to draft.

    ~/venvs/vllm-metis/bin/python tools/draft/regenerate.py TARGET_DIR PROMPTS_JSONL OUT_JSONL
        [--max-tokens 2048] [--thinking 0.2] [--max-len 4096] [--chunk 4096] [--limit N]

Writes one line per conversation, the features command's input (cli/features.cpp):
{"id": ..., "tokens": prompt + answer ids, "mask": 0 over the prompt, 1 over the answer}.
Sampling is the target's own generation config (Praxis: temperature 0.6, top-p 0.95, top-k 20),
the distribution users get. A --thinking fraction of conversations (chosen by id, so a rerun picks
the same ones) answer with thinking on, the rest with it off. Resumable: ids already in OUT_JSONL
are skipped, and answers are appended a chunk at a time. OUT_JSONL.done marks a finished run
(conversations too long for --max-len are left out).

The MoE runs on vLLM's Triton kernels by default: the FlashInfer CUTLASS backend vLLM picks on
sm_120 JIT-compiles on first use with one cicc per core at 3 to 7 GB each, which ran the shared
server out of memory (2026-10-24: the OOM killer took the CI runners). Any JIT build that still
happens is held to MAX_JOBS=2. tools/draft/pipeline.sh also runs this inside a memory-capped scope.
"""

import argparse
import hashlib
import json
import os
import pathlib
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target_dir")
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-len", type=int, default=4096, help="prompt + answer cap (vLLM max_model_len)")
    ap.add_argument("--thinking", type=float, default=0.2)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--max-seqs", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-memory", type=float, default=0.85)
    ap.add_argument("--eager", action="store_true", help="no torch.compile or CUDA graphs (starts fast, decodes slower)")
    ap.add_argument("--moe-backend", default="triton", help="vLLM's MoE kernels (see the module docstring)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # the first start compiles the model (minutes; cached afterwards), past vLLM's 600 s default
    os.environ.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "3600")
    os.environ.setdefault("MAX_JOBS", "2")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    target = pathlib.Path(args.target_dir)
    gen_cfg = json.loads((target / "generation_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(str(target), trust_remote_code=True)
    out = pathlib.Path(args.out)
    done = set()
    if out.exists():
        with open(out) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["id"])
    todo = []
    with open(args.prompts) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] not in done:
                todo.append(r)
            if args.limit and len(todo) + len(done) >= args.limit:
                break
    print(f"{len(done)} answered already, {len(todo)} to go", flush=True)
    finished = pathlib.Path(str(out) + ".done")
    if not todo:
        finished.touch()
        return

    def thinking(rid):
        h = int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return h < args.thinking

    def encode(r):
        text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=thinking(r["id"]))
        return tok(text, add_special_tokens=False)["input_ids"]

    llm = LLM(model=str(target), trust_remote_code=True, max_model_len=args.max_len,
              gpu_memory_utilization=args.gpu_memory, max_num_seqs=args.max_seqs, seed=args.seed,
              enforce_eager=args.eager, moe_backend=args.moe_backend)

    def params(prompt_len):
        return SamplingParams(temperature=gen_cfg.get("temperature", 0.6), top_p=gen_cfg.get("top_p", 0.95),
                              top_k=gen_cfg.get("top_k", 20), seed=args.seed,
                              max_tokens=min(args.max_tokens, args.max_len - prompt_len))

    t0, produced, answered = time.time(), 0, 0
    with open(out, "a") as f:
        for c in range(0, len(todo), args.chunk):
            batch, ids = [], []
            for r in todo[c : c + args.chunk]:
                p = encode(r)
                if len(p) + 16 <= args.max_len:  # leave room for at least a short answer
                    batch.append(p)
                    ids.append(r)
            outs = llm.generate([{"prompt_token_ids": p} for p in batch], [params(len(p)) for p in batch])
            for r, p, o in zip(ids, batch, outs):
                ans = list(o.outputs[0].token_ids)
                f.write(json.dumps({"id": r["id"], "tokens": p + ans, "mask": [0] * len(p) + [1] * len(ans)}) + "\n")
                produced += len(ans)
                answered += 1
            f.flush()
            dt = time.time() - t0
            print(f"{answered}/{len(todo)} answered, {produced} answer tokens, {produced / dt:.0f} tok/s, "
                  f"{produced / max(answered, 1):.0f} tokens per answer", flush=True)
    finished.touch()


if __name__ == "__main__":
    main()
