"""Speculation keeps logprobs: a lone greedy stream served with speculation (the default) reports the same tokens and
the same log-probabilities, bit for bit, as the same stream served with --no-spec. The two servers run one after the
other, so only one model is resident at a time.

  LX_BIN=build/dromos LX_MODEL=/path/to/model pytest -q tests/server/test_logprobs_spec.py
"""

import os
import socket
import subprocess
import time

import pytest
import requests

LX_BIN = os.environ.get("LX_BIN", "build/dromos")
LX_MODEL = os.environ["LX_MODEL"]
CHATS = [
    [{"role": "user", "content": "Explain why the sky is blue in two sentences."}],
    [{"role": "user", "content": "Write a Python function that reverses a list."}],
    [{"role": "user", "content": "List three prime numbers larger than 100."}],
]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def collect(extra):
    """Serves one model with the extra flags, asks every chat greedily with logprobs, and stops the server. Returns the
    replies with their logprobs, and the server's speculation counters after them (verify steps, drafts accepted)."""
    port = free_port()
    proc = subprocess.Popen(
        [LX_BIN, "serve", LX_MODEL, "--port", str(port), "--name", "praxis", "--sequences", "8", *extra],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(300):
            try:
                if requests.get(base + "/health", timeout=1).ok:
                    break
            except requests.RequestException:
                time.sleep(0.2)
        else:
            raise RuntimeError("the server did not start: " + proc.stdout.read())
        out = []
        for messages in CHATS:
            r = requests.post(
                base + "/v1/chat/completions",
                json={
                    "model": "praxis",
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 96,
                    "logprobs": True,
                    "top_logprobs": 3,
                },
                timeout=600,
            )
            r.raise_for_status()
            choice = r.json()["choices"][0]
            out.append((choice["message"]["content"], choice["logprobs"]["content"]))
        stats = requests.get(base + "/health", timeout=10).json()["speculation"]
        return out, stats
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise AssertionError("the server did not stop on SIGTERM")


def test_greedy_logprobs_match_without_speculation():
    with_spec, stats_on = collect([])
    without, stats_off = collect(["--no-spec"])
    # The comparison means something only if the speculating server really verified steps (and took drafts) while the
    # logprobs were asked for, and the other server did not.
    assert stats_on["verify_steps"] > 0 and stats_on["accepted"] > 0, stats_on
    assert stats_off["verify_steps"] == 0, stats_off
    assert len(with_spec) == len(without) == len(CHATS)
    for (text_on, lp_on), (text_off, lp_off) in zip(with_spec, without):
        assert text_on == text_off
        # Token, log-probability, bytes and the top list of every position, exactly.
        assert lp_on == lp_off
