#!/usr/bin/env python3
"""Chat template goldens from transformers, for the C++ render_chat tests.

    $PY golden_chat.py OUT_DIR NAME=MODEL_DIR [NAME=MODEL_DIR ...]

For each model folder writes OUT_DIR/NAME.jsonl: a header line {"model_dir", "transformers"} and one
case per line. The case's messages are in the neutral form the C++ side builds Messages from:
  {"role", "content", "reasoning"?, "tool_calls"?: [{"id"?, "name", "arguments": {...}}],
   "tool_call_id"?, "name"?}
and are converted here to transformers' message dicts exactly as the C++ side converts them (key
order included). "expected" is apply_chat_template(tokenize=False) for a folder with a tokenizer, and
render_jinja_template with the special tokens transformers passes otherwise. A rendering error is
recorded as "error" and the C++ side must throw.
"""
import json
import sys
from pathlib import Path

import transformers
from transformers.utils.chat_template_utils import render_jinja_template

SPECIAL_KEYS = ["bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"]

TOOLS_A = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the weather for a city. Units are \"C\" or \"F\".",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "days": {"type": "integer", "minimum": 1},
            "units": {"type": "string", "enum": ["C", "F"]},
        }, "required": ["city"]},
    }},
    {"type": "function", "function": {
        "name": "run_code",
        "description": "Run code.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"},
            "language": {"type": "string"},
            "options": {"type": "object", "properties": {
                "timeout": {"type": "number"}, "sandbox": {"type": "boolean"}}},
        }, "required": ["code"]},
    }},
]
TOOLS_B = [
    {"type": "function", "function": {
        "name": "lookup",
        "description": "Look up a word. Unicode works: 名前 and é.",
        "parameters": {"type": "object", "properties": {"term": {"type": "string"}}},
    }},
]


def call(name, arguments, call_id=""):
    c = {"name": name, "arguments": arguments}
    if call_id:
        c["id"] = call_id
    return c


def asst(content, reasoning="", calls=None):
    m = {"role": "assistant", "content": content}
    if reasoning:
        m["reasoning"] = reasoning
    if calls:
        m["tool_calls"] = calls
    return m


def user(text):
    return {"role": "user", "content": text}


def system(text):
    return {"role": "system", "content": text}


def tool(text, name="", call_id=""):
    m = {"role": "tool", "content": text}
    if name:
        m["name"] = name
    if call_id:
        m["tool_call_id"] = call_id
    return m


# (name, messages, tools)
BASES = [
    ("plain_user", [user("Hello")], None),
    ("system_user", [system("You are helpful."), user("Hi")], None),
    ("developer_user", [{"role": "developer", "content": "Be brief."}, user("Hi")], None),
    ("empty_system", [system(""), user("Hi")], None),
    ("padded_system", [system("  \n You are terse. \t\n"), user("Hi")], None),
    ("multi_turn", [user("Q1"), asst("A1"), user("Q2")], None),
    ("old_reasoning_dropped", [user("Q1"), asst("A1", reasoning="old thoughts"), user("Q2")], None),
    ("current_tool_loop", [
        user("Weather in Paris?"),
        asst("", reasoning="Need the weather tool.", calls=[call("get_weather", {"city": "Paris", "days": 3})]),
        tool("Sunny, 21C", name="get_weather"),
        asst("It is sunny.", reasoning="Answer directly."),
    ], TOOLS_A),
    ("reasoning_trimmed", [user("Q"), asst("  Answer  ", reasoning="\n  padded thought \n")], None),
    ("tool_args_types", [user("Go"), asst("", calls=[
        call("run_code", {"code": "print(1)\n", "language": "python",
                          "options": {"timeout": 2.5, "sandbox": True}})])], TOOLS_A),
    ("tool_args_scalars", [user("x"), asst("", calls=[
        call("f", {"n": 1, "neg": -7, "f": 1.0, "small": 1e-7, "t": True, "fl": False, "nothing": None,
                   "s": "a\"b\\c", "third": 0.1})])], None),
    ("tool_args_nested", [user("x"), asst("", calls=[
        call("g", {"zeta": {"b": 1, "a": [1, "two", {"y": None, "x": False}]},
                   "alpha": "é ☃", "list": [[], {}]})])], None),
    ("tool_args_newlines", [user("x"), asst("", calls=[
        call("note", {"text": "line1\nline2\t\"quoted\"", "path": "C:\\dir\\f"})])], None),
    ("multi_calls", [user("x"), asst("Two lookups.", calls=[
        call("lookup", {"term": "a"}), call("lookup", {"term": "b"})])], TOOLS_B),
    ("empty_args", [user("x"), asst("", calls=[call("ping", {})])], None),
    ("tool_result_named", [user("x"), asst("", calls=[call("get_weather", {"city": "Oslo"}, "call_1")]),
                           tool("{\"temp\": 3}", name="get_weather", call_id="call_1")], TOOLS_A),
    ("orphan_tool_result", [tool("orphan result")], None),
    ("assistant_first", [asst("Hello there")], None),
    ("system_only", [system("Only system.")], None),
    ("no_messages", [], None),
    ("user_special_text", [user("Literal <|fim_suffix|> and <think> text")], None),
    ("user_whitespace", [user("\n  indented\n\ttabbed  \n")], None),
    ("user_unicode", [user("日本語 emoji 🚀 café ‘quotes’ \\n backslash \"dq\"")], None),
    ("assistant_padded", [user("Q"), asst("\n\n  padded answer \n")], None),
    ("system_with_tools", [system("Use tools."), user("Weather?")], TOOLS_A),
    ("tools_no_system", [user("Weather?")], TOOLS_A),
    ("tools_unicode", [user("Define 名前")], TOOLS_B),
    ("tools_empty_list", [user("Hi")], []),
    ("tool_loop_two_rounds", [
        user("Q"),
        asst("", reasoning="r1", calls=[call("lookup", {"term": "a"})]),
        tool("res1"),
        asst("", reasoning="r2", calls=[call("lookup", {"term": "b"})]),
        tool("res2"),
        asst("done"),
    ], TOOLS_B),
    ("history_then_tool_loop", [
        user("old"), asst("old answer", reasoning="stale"), user("new"),
        asst("", reasoning="fresh", calls=[call("lookup", {"term": "x"})]), tool("r"), asst("final"),
    ], TOOLS_B),
    ("system_multiline", [system("Line one.\nLine two with ünïcode."), user("ok")], None),
    ("reasoning_only", [user("q"), asst("", reasoning="thinking only")], None),
    ("empty_reasoning_active", [user("q"), asst("answer")], None),
    ("argument_key_order", [user("x"), asst("", calls=[call("m", {"b": 1, "a": 2, "B": 3, "é": 4})])], None),
    ("empty_string_args", [user("x"), asst("", calls=[call("m", {"empty": "", "space": " "})])], None),
    ("long_conversation", [
        system("Be concise."), user("one"), asst("first", reasoning="old"), user("two"),
        asst("second"), user("three"), asst("third", reasoning="kept? no"), user("four"),
    ], None),
    ("tool_message_name_only", [user("x"), asst("", calls=[call("lookup", {"term": "z"})]),
                                tool("ok", name="lookup")], TOOLS_B),
    ("empty_user", [user("")], None),
    ("system_not_first", [user("hi"), system("late system")], None),
    ("markup_in_content", [user("q"), asst("<function=x>no</function> <|tool_call|> text")], None),
    ("developer_with_tools", [{"role": "developer", "content": "Dev rules"}, user("Q")], TOOLS_A),
    ("tool_args_order_unicode_keys", [user("x"), asst("", calls=[call("m", {"ü": 1, "a": [True, False, None]})])],
     None),
    ("assistant_tool_content_spaces", [user("x"), asst("   ", calls=[call("lookup", {"term": "q"})])], TOOLS_B),
]


def to_hf(message):
    """The transformers message dict, built the way the C++ side builds it (key order included)."""
    out = {"role": message["role"], "content": message.get("content", "")}
    if message.get("reasoning"):
        out["reasoning_content"] = message["reasoning"]
    if message.get("tool_calls"):
        calls = []
        for c in message["tool_calls"]:
            entry = {}
            if c.get("id"):
                entry["id"] = c["id"]
            entry["type"] = "function"
            entry["function"] = {"name": c["name"], "arguments": c["arguments"]}
            calls.append(entry)
        out["tool_calls"] = calls
    if message.get("tool_call_id"):
        out["tool_call_id"] = message["tool_call_id"]
    if message.get("name"):
        out["name"] = message["name"]
    return out


def special_tokens(model_dir):
    cfg_path = Path(model_dir) / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    out = {}
    for key in SPECIAL_KEYS:
        value = cfg.get(key)
        if isinstance(value, dict):
            value = value.get("content")
        if isinstance(value, str):
            out[key] = value
    extra = cfg.get("additional_special_tokens")
    if isinstance(extra, list):
        out["additional_special_tokens"] = [t for t in extra if isinstance(t, str)]
    return out


def template_source(model_dir):
    path = Path(model_dir) / "chat_template.jinja"
    if path.exists():
        return path.read_text(encoding="utf-8")
    cfg = json.loads((Path(model_dir) / "tokenizer_config.json").read_text(encoding="utf-8"))
    chat = cfg.get("chat_template")
    if isinstance(chat, str):
        return chat
    if isinstance(chat, list):
        for entry in chat:
            if entry.get("name") == "default":
                return entry["template"]
    raise SystemExit(f"no chat template in {model_dir}")


def cases():
    for name, messages, tools in BASES:
        for thinking in (True, False):
            for agp in (True, False):
                yield name, messages, tools, thinking, agp


def render_fn(model_dir):
    """apply_chat_template when the folder loads as a tokenizer, else transformers' renderer."""
    tpl = template_source(model_dir)
    specials = special_tokens(model_dir)
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        tok = None

    def render(messages, tools, thinking, agp):
        hf_messages = [to_hf(m) for m in messages]
        if not hf_messages:
            # apply_chat_template's own check; the fallback renderer below skips it.
            raise ValueError("Cannot apply chat template to an empty conversation. Provide at least one message.")
        if tok is not None:
            return tok.apply_chat_template(hf_messages, tools=tools, add_generation_prompt=agp,
                                           tokenize=False, enable_thinking=thinking)
        out = render_jinja_template(conversations=[hf_messages], tools=tools, documents=None,
                                    chat_template=tpl, add_generation_prompt=agp,
                                    enable_thinking=thinking, **specials)
        # (rendered strings per conversation, generation indices)
        if isinstance(out, tuple):
            out = out[0]
        return out[0] if isinstance(out, list) else out

    return render, tok is not None


def neutral_case(name, messages, tools, thinking, agp):
    return {"name": name, "messages": messages, "tools": tools, "thinking": thinking,
            "add_generation_prompt": agp}


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for arg in sys.argv[2:]:
        name, model_dir = arg.split("=", 1)
        try:
            render, with_tokenizer = render_fn(model_dir)
        except (SystemExit, OSError, KeyError) as exc:
            print(json.dumps({"name": name, "skipped": str(exc)}))
            continue
        path = out_dir / f"{name}.jsonl"
        count = 0
        with open(path, "w", encoding="utf-8") as out:
            header = {"model_dir": model_dir, "transformers": transformers.__version__,
                      "tokenizer": with_tokenizer, "template_source": "chat_template.jinja" if (
                          Path(model_dir) / "chat_template.jinja").exists() else "tokenizer_config.json"}
            out.write(json.dumps(header, ensure_ascii=True) + "\n")
            for case_name, messages, tools, thinking, agp in cases():
                case = neutral_case(case_name, messages, tools, thinking, agp)
                try:
                    case["expected"] = render(messages, tools, thinking, agp)
                except Exception as exc:  # the C++ side must throw for these
                    case["error"] = f"{type(exc).__name__}: {exc}"
                out.write(json.dumps(case, ensure_ascii=True, separators=(",", ":")) + "\n")
                count += 1
        print(json.dumps({"name": name, "cases": count, "tokenizer": with_tokenizer}))


if __name__ == "__main__":
    main()
