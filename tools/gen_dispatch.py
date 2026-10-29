#!/usr/bin/env python3
"""Generates the backend dispatch table: the lx_* stubs host code calls, and the table they forward through.

    python3 tools/gen_dispatch.py OUT.cpp engine/cuda/kernels.h engine/cuda/tunable.h

Every LX_API function declared in the headers gets three things in OUT.cpp: a function-pointer slot in one
table, an extern "C" stub that forwards through the slot, and a missing function that holds the slot until
lx::dispatch::detail::bind() finds the symbol in the loaded backend library. A declaration it cannot parse, or
one that conflicts with an earlier declaration of the same name, is an error: the tool says where, removes OUT
and exits 1. On success it prints nothing.
"""

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The missing functions that return something other than the default. The default is LX_ERR_NOT_SUPPORTED for
# an int, 0 for other arithmetic, nullptr for a pointer, the text "backend lacks <name>" for a const char*, and
# nothing for void. The query functions below would read 801 as a status code, so they get their own values.
OVERRIDES = {
    "lx_rt_device_count": "0",
    "lx_rt_backend": "-1",
    "lx_arch": "0",
    "lx_rt_last_error": "0",
    "lx_rt_is_host_memory": "0",
    "lx_rt_process_memory": "LX_ERR_NOT_SUPPORTED",
    "lx_tune_get": "-1",
    "lx_tune_variants": "0",
    "lx_tune_kernels": "0",
    "lx_tune_runnable": "0",
    "lx_tune_key": "-1",
    "lx_tune_run": "LX_ERR_NOT_SUPPORTED",
}

# Words that are types, never parameter names.
TYPE_WORDS = {
    "void", "char", "short", "int", "long", "float", "double", "bool", "signed", "unsigned", "const", "volatile",
    "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "size_t",
}

# Return types whose missing value is 0 (int has its own default).
ZERO_RETURNS = {
    "char", "short", "long", "float", "double", "bool", "signed", "unsigned", "int8_t", "int16_t", "int32_t",
    "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "size_t",
}


class ParseError(Exception):
    pass


def fail(where, message):
    raise ParseError("%s: %s" % (where, message))


def label(path):
    rel = os.path.relpath(os.path.abspath(path), REPO)
    return os.path.basename(path) if rel.startswith("..") else rel


def clean(path, text):
    """The header with comments blanked and preprocessor lines removed. Line numbers are kept."""
    out = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i) or text.startswith("/*", i):
            if text.startswith("//", i):
                end = text.find("\n", i)
                end = n if end < 0 else end
            else:
                end = text.find("*/", i + 2)
                if end < 0:
                    fail("%s:%d" % (path, text.count("\n", 0, i) + 1), "unterminated /* comment")
                end += 2
            out.append("".join("\n" if ch == "\n" else " " for ch in text[i:end]))
            i = end
        elif text[i] in "\"'":
            end = i + 1
            while end < n and text[end] != text[i] and text[end] != "\n":
                end += 2 if text[end] == "\\" else 1
            if end >= n or text[end] != text[i]:
                fail("%s:%d" % (path, text.count("\n", 0, i) + 1), "unterminated literal")
            out.append(text[i:end + 1])
            i = end + 1
        else:
            out.append(text[i])
            i += 1
    lines = "".join(out).split("\n")
    continued = False
    for k, line in enumerate(lines):
        if continued or line.lstrip().startswith("#"):
            continued = line.rstrip().endswith("\\")
            lines[k] = ""
    return "\n".join(lines)


def normalize_type(text):
    text = " ".join(text.split())
    return re.sub(r"\s*\*", "*", text)


def typedef_names(code):
    names = set(re.findall(r"\}\s*([A-Za-z_]\w*)\s*;", code))
    names |= set(re.findall(r"\btypedef\b[^{};]*?\b([A-Za-z_]\w*)\s*;", code))
    return names


def parse_param(where, raw, index, typedefs):
    """(type, name) of one parameter. An unnamed parameter gets the synthetic name a<index>."""
    text = " ".join(raw.split())
    if not re.fullmatch(r"[A-Za-z_][\w\s*]*", text):
        fail(where, "cannot parse parameter '%s'" % raw.strip())
    words = re.findall(r"[A-Za-z_]\w*|\*", text)
    last = words[-1]
    if last != "*" and last not in TYPE_WORDS and last not in typedefs:
        if len(words) < 2:
            fail(where, "unknown type '%s' in parameter '%s'" % (last, text))
        return normalize_type(text[: len(text) - len(last)]), last
    return normalize_type(text), "a%d" % index


def declarations(path, code, typedefs):
    found = []
    for m in re.finditer(r"\bLX_API\b", code):
        where = "%s:%d" % (path, code.count("\n", 0, m.start()) + 1)
        end = code.find(";", m.end())
        if end < 0:
            fail(where, "LX_API declaration has no ';'")
        body = " ".join(code[m.end():end].split())
        if (body.count("(") != 1 or body.count(")") != 1 or not body.endswith(")")
                or re.search(r"[{}=\[\]]", body)):
            fail(where, "cannot parse LX_API declaration '%s' (one parameter list, no nesting)" % body)
        head, _, tail = body.partition("(")
        match = re.fullmatch(r"(.+?)\s*\b([A-Za-z_]\w*)", head.strip())
        if not match:
            fail(where, "no return type and name in '%s'" % head.strip())
        ret = normalize_type(match.group(1))
        name = match.group(2)
        if not re.fullmatch(r"[A-Za-z_][\w ]*\**", ret):
            fail(where, "unsupported return type '%s' of %s" % (ret, name))
        params_text = tail[:-1].strip()
        params = []
        if params_text not in ("", "void"):
            for index, raw in enumerate(params_text.split(",")):
                params.append(parse_param(where, raw, index, typedefs))
        found.append({"name": name, "ret": ret, "params": params, "where": where})
    return found


def signature(decl):
    return decl["ret"], tuple(t for t, _ in decl["params"])


def param_list(decl):
    return ", ".join("%s %s" % (t, n) for t, n in decl["params"]) or "void"


def type_list(decl):
    return ", ".join(t for t, _ in decl["params"]) or "void"


def arg_list(decl):
    return ", ".join(n for _, n in decl["params"])


def missing_value(decl):
    ret, name = decl["ret"], decl["name"]
    if ret == "void":
        if name in OVERRIDES:
            fail(decl["where"], "%s returns nothing, so it takes no missing value" % name)
        return None
    if name in OVERRIDES:
        return OVERRIDES[name]
    if ret == "const char*":
        return '"backend lacks %s"' % name
    if ret.endswith("*"):
        return "nullptr"
    if ret == "int":
        return "LX_ERR_NOT_SUPPORTED"
    if ret in ZERO_RETURNS:
        return "0"
    fail(decl["where"], "no missing value for return type '%s' of %s" % (ret, name))


TABLE_HEAD = """
// Loads the backend on first use, then returns the table. The function-local static runs ensure() once, and
// after that a call costs the guard check and one pointer load.
Table& dispatch_table() {
  static const bool loaded = [] {
    ::lx::dispatch::detail::ensure();
    return true;
  }();
  (void)loaded;
  return g_table;
}
"""

SLOT_TEMPLATE = """
// Points the slot at the backend's symbol, or at its missing function (and records the name) when the backend
// does not export it.
template <typename Fn>
void slot(void* lib, const char* name, Fn& fn, Fn missing, std::vector<std::string>& absent) {
  if (void* sym = dlsym(lib, name)) {
    fn = reinterpret_cast<Fn>(sym);
  } else {
    fn = missing;
    absent.push_back(name);
  }
}
"""


def emit(decls, paths):
    o = [
        "// Generated by tools/gen_dispatch.py from %s." % ", ".join(label(p) for p in paths),
        "// Do not edit: change the headers or the generator, then build again.",
        "//",
        "// Each LX_API function in those headers has three parts here:",
        "//   a slot in Table, holding a function pointer to the backend's implementation;",
        '//   an extern "C" stub with the same name and signature, which host code links against and which',
        "//     calls through the slot;",
        "//   a missing function, which the slot holds until bind() finds the symbol in the loaded backend. It",
        "//     reports the symbol (lx::dispatch::report_missing) and returns a neutral value.",
        "// Every slot starts on its missing function, so a call before any backend loads is safe.",
        "",
        '#include "cuda/kernels.h"',
        '#include "cuda/tunable.h"',
        '#include "dispatch/dispatch_internal.h"',
        "",
        "#include \"platform/posix.h\"",
        "",
        "#include <string>",
        "#include <vector>",
        "",
        "namespace {",
        "",
        "// One slot per LX_API function, in header order.",
        "struct Table {",
    ]
    for d in decls:
        o.append("  %s (*%s)(%s);" % (d["ret"], d["name"], param_list(d)))
    o += ["};", "", "// The missing functions: report the symbol, then return a neutral value.", ""]
    for d in decls:
        o.append("%s missing_%s(%s) {" % (d["ret"], d["name"], type_list(d)))
        o.append('  ::lx::dispatch::report_missing("%s");' % d["name"])
        value = missing_value(d)
        if value is not None:
            o.append("  return %s;" % value)
        o += ["}", ""]
    o += [
        "// Every slot starts on its missing function. The table is constant-initialised, so it is valid before",
        "// any code runs.",
        "Table g_table = {",
    ]
    for d in decls:
        o.append("    .%s = &missing_%s," % (d["name"], d["name"]))
    o += ["};", ""]
    o += SLOT_TEMPLATE.split("\n")
    o += TABLE_HEAD.split("\n")
    o += ["}  // namespace", "", "namespace lx::dispatch::detail {", "",
          "void bind(void* lib, std::vector<std::string>& missing) {"]
    for d in decls:
        o.append('  slot(lib, "%s", g_table.%s, &missing_%s, missing);' % (d["name"], d["name"], d["name"]))
    o += ["}", "", "}  // namespace lx::dispatch::detail", "", 'extern "C" {', ""]
    for d in decls:
        if d["ret"] == "void":
            o.append("void %s(%s) {" % (d["name"], param_list(d)))
            o.append("  dispatch_table().%s(%s);" % (d["name"], arg_list(d)))
        else:
            o.append("%s %s(%s) {" % (d["ret"], d["name"], param_list(d)))
            o.append("  return dispatch_table().%s(%s);" % (d["name"], arg_list(d)))
        o += ["}", ""]
    o += ['}  // extern "C"', ""]
    return "\n".join(o)


def generate(paths):
    sources = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            sources.append((path, clean(path, f.read())))
    typedefs = set()
    for _, code in sources:
        typedefs |= typedef_names(code)
    symbols = {}
    for path, code in sources:
        for decl in declarations(path, code, typedefs):
            prior = symbols.get(decl["name"])
            if prior is None:
                symbols[decl["name"]] = decl
            elif signature(prior) != signature(decl):
                fail(decl["where"], "%s is declared differently than at %s" % (decl["name"], prior["where"]))
            # An identical redeclaration (kernels.h declares lx_dequantize_weight twice) is one slot.
    for name in OVERRIDES:
        if name not in symbols:
            fail("tools/gen_dispatch.py", "missing-value override for %s, which no header declares" % name)
    return emit(list(symbols.values()), paths)


def main(argv):
    if len(argv) < 3:
        sys.stderr.write("usage: python3 tools/gen_dispatch.py OUT.cpp kernels.h [tunable.h ...]\n")
        return 2
    out_path, headers = argv[1], argv[2:]
    try:
        text = generate(headers)
    except (ParseError, OSError) as e:
        sys.stderr.write("gen_dispatch.py: error: %s\n" % e)
        try:
            os.remove(out_path)
        except OSError:
            pass
        return 1
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
