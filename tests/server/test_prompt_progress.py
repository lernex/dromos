"""Prompt progress: a streamed request with return_progress hears its prompt read before its first token, in order and
never past what the device has done; the same prompt again starts from what the prefix cache holds; a request without
the flag sees none of it.

  LX_BIN=build/dromos LX_MODEL=/path/to/model pytest -q tests/server/test_prompt_progress.py
"""

import json
import os
import socket
import subprocess
import time

import pytest
import requests

LX_BIN = os.environ.get("LX_BIN", "build/dromos")
LX_MODEL = os.environ["LX_MODEL"]

# Long enough to take several prefill steps on the 512-token grid.
PAPER = " ".join(f"Paragraph {i}: the river bends east past the mill, then north toward the old stone bridge." for i in range(220))
MESSAGES = [{"role": "user", "content": PAPER + "\n\nIn one sentence, where does the river go?"}]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def base():
    port = free_port()
    proc = subprocess.Popen(
        [LX_BIN, "serve", LX_MODEL, "--port", str(port), "--name", "praxis", "--sequences", "2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(300):
            try:
                if requests.get(url + "/health", timeout=1).ok:
                    break
            except requests.RequestException:
                time.sleep(0.2)
        else:
            raise RuntimeError("the server did not start: " + proc.stdout.read())
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=60)


def chunks(base, progress):
    body = {"model": "praxis", "messages": MESSAGES, "temperature": 0, "max_tokens": 24, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    if progress:
        body["return_progress"] = True
    with requests.post(base + "/v1/chat/completions", json=body, stream=True, timeout=600) as r:
        assert r.status_code == 200, r.text
        out = []
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            out.append(json.loads(payload))
        return out


def reading(stream):
    return [c["prompt_progress"] for c in stream if "prompt_progress" in c]


def first_token_at(stream):
    for i, c in enumerate(stream):
        for choice in c.get("choices", []):
            d = choice.get("delta", {})
            if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                return i
    return len(stream)


def test_progress_reads_in_order_before_the_first_token(base):
    stream = chunks(base, progress=True)
    seen = reading(stream)
    assert len(seen) >= 3, seen
    total = seen[0]["total"]
    assert total > 1024
    assert all(p["total"] == total for p in seen)
    assert seen[0]["processed"] == seen[0]["cache"]
    assert seen[-1]["processed"] == total
    processed = [p["processed"] for p in seen]
    assert processed == sorted(processed)
    assert [p["time_ms"] for p in seen] == sorted(p["time_ms"] for p in seen)
    # Every word on the reading comes before the tokens it led to, and every chunk of it has no choices.
    last_progress = max(i for i, c in enumerate(stream) if "prompt_progress" in c)
    assert last_progress < first_token_at(stream)
    assert all(c["choices"] == [] for c in stream if "prompt_progress" in c)


def test_the_same_prompt_again_starts_from_the_cache(base):
    chunks(base, progress=True)
    seen = reading(chunks(base, progress=True))
    assert seen[0]["cache"] > 0, seen[0]
    assert seen[0]["processed"] == seen[0]["cache"]
    assert seen[-1]["processed"] == seen[-1]["total"]


def test_no_progress_unless_asked(base):
    assert reading(chunks(base, progress=False)) == []
