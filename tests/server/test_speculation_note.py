"""serve says once, at startup, whether speculation is on and why it is not: --no-spec names the flag, and a speculating
server says what it does (a lone stream verifies drafts). The offloaded case is doctor's to say; this checks serve's line.

  LX_BIN=build/dromos LX_MODEL=/path/to/model pytest -q tests/server/test_speculation_note.py
"""

import os
import socket
import subprocess
import time

import requests

LX_BIN = os.environ.get("LX_BIN", "build/dromos")
LX_MODEL = os.environ["LX_MODEL"]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def startup_output(extra):
    port = free_port()
    proc = subprocess.Popen(
        [LX_BIN, "serve", LX_MODEL, "--port", str(port), "--name", "praxis", "--sequences", "2", *extra],
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
    finally:
        proc.terminate()
    out, _ = proc.communicate(timeout=30)
    return out


def test_no_spec_says_why():
    out = startup_output(["--no-spec"])
    assert out.count("speculation off: --no-spec") == 1, out


def test_speculating_server_says_what_it_does_once():
    out = startup_output([])
    assert out.count("speculation on: a lone stream verifies drafts") == 1, out
    assert "speculation off" not in out, out
