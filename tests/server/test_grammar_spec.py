"""Speculation under a grammar: a lone greedy stream whose requests carry tools (the tool grammar constrains its reply)
gives the same reply served with speculation (the default) as with --no-spec, and the speculating server really verified
steps and took drafts while it did. The two servers run one after the other, so one model is resident at a time.

  LX_BIN=build/dromos LX_MODEL=/path/to/model pytest -q tests/server/test_grammar_spec.py
"""

import os
import socket
import subprocess
import time

import pytest
import requests

LX_BIN = os.environ.get("LX_BIN", "build/dromos")
LX_MODEL = os.environ["LX_MODEL"]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the local time in a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    },
]
REQUESTS = [
    [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}],
    [{"role": "user", "content": "What is the time in Tokyo and in Lima? Use the tools."}],
]


def strip_ids(message):
    out = dict(message)
    out["tool_calls"] = [{k: v for k, v in call.items() if k != "id"} for call in message.get("tool_calls") or []]
    return out


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def collect(extra):
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
        replies = []
        for messages in REQUESTS:
            r = requests.post(
                base + "/v1/chat/completions",
                json={"model": "praxis", "messages": messages, "tools": TOOLS, "max_tokens": 600, "temperature": 0},
                timeout=600,
            )
            r.raise_for_status()
            choice = r.json()["choices"][0]
            replies.append((choice["finish_reason"], choice["message"]))
        stats = requests.get(base + "/health", timeout=10).json()["speculation"]
        return replies, stats
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise AssertionError("the server did not stop on SIGTERM")


def test_tool_replies_match_without_speculation():
    with_spec, stats_on = collect([])
    without, stats_off = collect(["--no-spec"])
    assert stats_on["verify_steps"] > 0 and stats_on["accepted"] > 0, stats_on
    assert stats_off["verify_steps"] == 0, stats_off
    assert len(with_spec) == len(without) == len(REQUESTS)
    for (finish_on, msg_on), (finish_off, msg_off) in zip(with_spec, without):
        assert finish_on == finish_off == "tool_calls"
        # Call ids are random per request; everything else is the reply, exactly.
        assert strip_ids(msg_on) == strip_ids(msg_off)
