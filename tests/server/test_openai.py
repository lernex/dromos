"""The OpenAI-compatible server, end to end, with the official openai client and plain HTTP.

  LX_BIN=build/lernex LX_MODEL=/path/to/model test_openai.py

Starts `lernex serve` on a free port and checks the wire format (streaming and not), stop strings,
logprobs, usage, tool requests, errors, concurrent requests, and that a client that disconnects
mid-stream does not hold the engine. Needs the GPU, so run it under scripts/remote.sh ... x.
"""

import json
import os
import socket
import subprocess
import sys
import time

import openai
import pytest
import requests

LX_BIN = os.environ.get("LX_BIN", "build/lernex")
LX_MODEL = os.environ["LX_MODEL"]
# Greedy from these ids the Praxis checkpoint continues EXPECTED: the engine's tokens
# [205, 205, 534, 1726, 324, 8232] agree with the reference argmax at every step (tests/gpu).
PROMPT = [1001, 2002, 3003]
EXPECTED = "\n\nThe user is asking"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = free_port()
    proc = subprocess.Popen(
        [LX_BIN, "serve", LX_MODEL, "--port", str(port), "--name", "praxis", "--sequences", "8"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(300):
        try:
            if requests.get(base + "/health", timeout=1).ok:
                break
        except requests.RequestException:
            time.sleep(0.2)
    else:
        proc.kill()
        raise RuntimeError("the server did not start: " + proc.stdout.read())
    yield base, port
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise AssertionError("the server did not stop on SIGTERM")


@pytest.fixture(scope="module")
def client(server):
    base, _ = server
    return openai.OpenAI(base_url=base + "/v1", api_key="unused")


def test_health_and_models(server, client):
    base, _ = server
    assert requests.get(base + "/health").json()["status"] == "ok"
    models = client.models.list()
    assert [m.id for m in models.data] == ["praxis"]


def test_completion_with_token_ids(client):
    r = client.completions.create(model="praxis", prompt=PROMPT, max_tokens=6, temperature=0)
    assert r.choices[0].text == EXPECTED
    assert r.choices[0].finish_reason == "length"
    assert r.usage.prompt_tokens == 3 and r.usage.completion_tokens == 6


def test_completion_stream_matches_and_ends_with_usage(client):
    chunks = list(
        client.completions.create(
            model="praxis",
            prompt=PROMPT,
            max_tokens=6,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    text = "".join(c.choices[0].text for c in chunks if c.choices)
    assert text == EXPECTED
    assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens == 6


def test_stop_string_ends_the_reply_before_it(client):
    r = client.completions.create(model="praxis", prompt=PROMPT, max_tokens=20, temperature=0, stop=["asking"])
    assert r.choices[0].text == "\n\nThe user is "
    assert r.choices[0].finish_reason == "stop"


def test_chat_non_streaming(client):
    r = client.chat.completions.create(
        model="praxis",
        messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}],
        max_tokens=5,
        temperature=0,
    )
    message = r.choices[0].message
    assert message.role == "assistant" and isinstance(message.content, str)
    assert r.choices[0].finish_reason == "length"
    assert r.usage.completion_tokens == 5


def test_chat_streaming_with_usage(client):
    stream = client.chat.completions.create(
        model="praxis",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=4,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = list(stream)
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[-1].usage.completion_tokens == 4
    finish = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
    assert finish == ["length"]


def test_chat_logprobs(client):
    r = client.chat.completions.create(
        model="praxis",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=3,
        temperature=0,
        logprobs=True,
        top_logprobs=2,
    )
    content = r.choices[0].logprobs.content
    assert len(content) == 3
    assert all(len(t.top_logprobs) == 2 for t in content)
    assert all(t.logprob <= 0 for t in content)


def test_tools_and_tool_results_are_accepted(client):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    messages = [
        {"role": "user", "content": "Weather in Oslo?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Oslo\"}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "4 C, rain"},
    ]
    r = client.chat.completions.create(model="praxis", messages=messages, tools=tools, max_tokens=4, temperature=0)
    assert r.choices[0].message.role == "assistant"
    named = client.chat.completions.create(
        model="praxis",
        messages=[{"role": "user", "content": "Weather?"}],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        max_tokens=2,
        temperature=0,
    )
    assert named.choices[0].finish_reason in ("length", "stop", "tool_calls")
    none = client.chat.completions.create(
        model="praxis",
        messages=[{"role": "user", "content": "Weather?"}],
        tools=tools,
        tool_choice="none",
        max_tokens=2,
        temperature=0,
    )
    assert none.choices[0].message.tool_calls is None


def test_tool_round_trip(client):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }
    ]
    messages = [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]
    first = client.chat.completions.create(model="praxis", messages=messages, tools=tools, max_tokens=600, temperature=0)
    assert first.choices[0].finish_reason == "tool_calls"
    call = first.choices[0].message.tool_calls[0]
    assert call.function.name == "get_weather" and json.loads(call.function.arguments)["city"] == "Paris"
    messages.append(
        {
            "role": "assistant",
            "content": first.choices[0].message.content or "",
            "tool_calls": [
                {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}}
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps({"city": "Paris", "temp_c": 18})})
    second = client.chat.completions.create(model="praxis", messages=messages, tools=tools, max_tokens=400, temperature=0)
    assert second.choices[0].finish_reason == "stop"
    assert "18" in (second.choices[0].message.content or "")


def test_errors_are_clear(server):
    base, _ = server
    url = base + "/v1/chat/completions"
    cases = [
        ({"messages": []}, 400, "messages"),
        ({"messages": [{"role": "user", "content": "x"}], "temperature": 5}, 400, "temperature"),
        ({"messages": [{"role": "user", "content": "x"}], "n": 2}, 400, "n"),
        ({"messages": [{"role": "user", "content": "x"}], "model": "gpt-9"}, 404, "model"),
        ({"messages": [{"role": "robot", "content": "x"}]}, 400, "messages"),
        ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}, 400, "messages"),
        ({"messages": [{"role": "user", "content": "x"}], "top_logprobs": 3}, 400, "top_logprobs"),
    ]
    for body, status, param in cases:
        r = requests.post(url, json=body)
        assert r.status_code == status, (body, r.text)
        assert r.json()["error"]["param"] == param, r.text
    r = requests.post(url, data="{bad json", headers={"content-type": "application/json"})
    assert r.status_code == 400 and "not valid JSON" in r.json()["error"]["message"]
    r = requests.post(base + "/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 1 << 40})
    assert r.status_code == 400
    assert requests.get(base + "/nope").status_code == 404
    too_long = [i % 60000 + 1 for i in range(100000)]  # longer than any context the plan allows
    r = requests.post(base + "/v1/completions", json={"prompt": too_long, "max_tokens": 2})
    assert r.status_code == 400 and "context" in r.json()["error"]["message"]


def test_parallel_requests_all_finish(client):
    import concurrent.futures

    def one(i):
        r = client.completions.create(model="praxis", prompt=[i, 5, 6], max_tokens=24, temperature=0.8, seed=i)
        return r.usage.completion_tokens

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        counts = list(pool.map(one, range(8)))
    assert counts == [24] * 8


def test_a_disconnected_stream_is_cancelled(server, client):
    base, _ = server
    body = {"prompt": PROMPT, "max_tokens": 2000, "temperature": 0.7, "stream": True, "seed": 9}
    r = requests.post(base + "/v1/completions", json=body, stream=True)
    seen = 0
    for line in r.iter_lines():
        if line.startswith(b"data:"):
            seen += 1
        if seen >= 3:
            break
    r.close()
    # the engine notices the closed socket and frees the slot; the next request still runs at once
    for _ in range(50):
        h = requests.get(base + "/health").json()
        if h["running"] == 0:
            break
        time.sleep(0.2)
    assert h["running"] == 0, h
    r = client.completions.create(model="praxis", prompt=PROMPT, max_tokens=6, temperature=0)
    assert r.choices[0].text == EXPECTED
