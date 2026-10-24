"""Conversation reuse and shared system prompts, measured through the server.

  prefix_bench.py conversation URL OUT.json [--turns N] [--think on|off]
      A long conversation, one turn after another, greedy. Each turn's time to first token and its
      full text go to OUT.json, so a run with the prefix cache and one without can be compared.
  prefix_bench.py shared URL OUT.json [--requests N]
      N requests with one long system prompt and different questions, one after another. Their
      time to first token shows what the shared prefix saves.

Greedy decoding is deterministic, so two runs of the same conversation must produce the same text.
"""

import json
import statistics
import sys
import time

import openai

SYSTEM = (
    "You are Lernex, a patient tutor. Teach with short, exact explanations and check understanding "
    "before moving on. Use plain language, one idea at a time, and worked examples where they help. "
    "Never invent facts; say when something is uncertain. Keep answers under two hundred words unless "
    "the learner asks for more. When a learner is wrong, name the exact step where the reasoning went "
    "off, then show the corrected step. Prefer the learner's own words when you restate a question. "
) * 12

TOPICS = [
    "why the sum of the first n odd numbers is n squared",
    "how a binary search decides which half to keep",
    "what a derivative measures, with the slope of a falling ball",
    "why photosynthesis needs light, and where the energy is stored",
    "the difference between a mean, a median and a mode, with a skewed example",
    "how compound interest grows a balance over ten years",
    "what makes a chemical bond ionic rather than covalent",
    "how a hash table handles two keys that land in the same bucket",
    "why the Earth has seasons even though it is closest to the sun in January",
    "how to solve a system of two linear equations by substitution",
    "what a recursive function needs before it can stop",
    "how the immune system remembers an infection it has seen before",
    "why water expands when it freezes",
    "how to read a p-value without over-reading it",
]


def turn_text(i):
    topic = TOPICS[i % len(TOPICS)]
    return (
        f"Question {i + 1}: I am studying {topic}. Could you walk me through it step by step, "
        f"with one small example I can check myself, and then ask me one question about it? "
        f"Please keep the example concrete and avoid jargon I have not seen before."
    )


def stream_reply(client, messages, think, max_tokens):
    start = time.perf_counter()
    first = None
    content, reasoning = [], []
    usage = None
    finish = None
    stream = client.chat.completions.create(
        model="praxis",
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": think}},
    )
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish = choice.finish_reason
        delta = choice.delta
        r = getattr(delta, "reasoning_content", None) or ""
        c = delta.content or ""
        if (r or c) and first is None:
            first = time.perf_counter() - start
        reasoning.append(r)
        content.append(c)
    total = time.perf_counter() - start
    return {
        "ttft": first if first is not None else total,
        "total": total,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "finish": finish,
        "reasoning": "".join(reasoning),
        "content": "".join(content),
    }


def conversation(url, out, turns, think):
    client = openai.OpenAI(base_url=url + "/v1", api_key="unused")
    messages = [{"role": "system", "content": SYSTEM}]
    rows = []
    for i in range(turns):
        messages.append({"role": "user", "content": turn_text(i)})
        row = stream_reply(client, messages, think, 96)
        row["turn"] = i + 1
        rows.append(row)
        print(
            f"turn {i + 1:2d}  prompt {row['prompt_tokens']:5d} tok  ttft {row['ttft'] * 1000:7.0f} ms  "
            f"total {row['total'] * 1000:6.0f} ms  {row['completion_tokens']} tok",
            flush=True,
        )
        messages.append({"role": "assistant", "content": row["content"]})
    with open(out, "w") as f:
        json.dump({"think": think, "turns": rows}, f, indent=1)


def shared(url, out, requests):
    client = openai.OpenAI(base_url=url + "/v1", api_key="unused")
    rows = []
    for i in range(requests):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": turn_text(i)}]
        row = stream_reply(client, messages, False, 48)
        row["request"] = i + 1
        rows.append(row)
        print(
            f"request {i + 1:2d}  prompt {row['prompt_tokens']:5d} tok  ttft {row['ttft'] * 1000:7.0f} ms",
            flush=True,
        )
    ttfts = [r["ttft"] for r in rows]
    summary = {
        "requests": requests,
        "ttft_first_ms": ttfts[0] * 1000,
        "ttft_p50_ms": statistics.median(ttfts) * 1000,
        "ttft_mean_ms": statistics.mean(ttfts) * 1000,
    }
    print(json.dumps(summary))
    with open(out, "w") as f:
        json.dump({"summary": summary, "requests": rows}, f, indent=1)


if __name__ == "__main__":
    mode, url, out = sys.argv[1], sys.argv[2], sys.argv[3]
    args = sys.argv[4:]

    def opt(name, default):
        return args[args.index(name) + 1] if name in args else default

    if mode == "conversation":
        conversation(url, out, int(opt("--turns", "12")), opt("--think", "on") == "on")
    elif mode == "shared":
        shared(url, out, int(opt("--requests", "16")))
    else:
        raise SystemExit("mode is conversation or shared")
