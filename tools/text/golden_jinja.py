#!/usr/bin/env python3
"""Jinja interpreter goldens: snippets rendered by jinja2 with transformers' environment.

    $PY golden_jinja.py OUT_DIR

Writes OUT_DIR/jinja.jsonl: one case per line {"name", "template", "vars", "expected"} or {..., "error"}.
The environment is the one transformers builds for chat templates (ImmutableSandboxedEnvironment with
trim_blocks, lstrip_blocks and the loopcontrols extension; tojson without HTML escaping; raise_exception
and strftime_now as globals).
"""
import json
import sys
from pathlib import Path

from jinja2 import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment


def make_env():
    import json as _json
    from jinja2 import nodes
    from jinja2.ext import Extension

    class AssistantTracker(Extension):
        # transformers' {% generation %} block: renders its body (the tracking is not needed here).
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
            return nodes.CallBlock(self.call_method("_generation_support"), [], [], body).set_lineno(lineno)

        def _generation_support(self, caller):
            return caller()

    def raise_exception(message):
        raise TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return _json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[AssistantTracker, loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


CASES = [
    # whitespace control
    ("ws_dash_both", "a  {%- if true -%}  b  {%- endif -%}  c", {}),
    ("ws_var_dash", "x  {{- 1 -}}  y", {}),
    ("lstrip_blocks", "one\n   {% if true %}two{% endif %}\nthree", {}),
    ("trim_blocks_newline", "a{% if true %}\nb{% endif %}\nc", {}),
    ("comment_trim", "a{# hidden #}\nb{#- also -#}  c", {}),
    ("plus_keeps_indent", "one\n   {%+ if true %}two{% endif %}", {}),
    ("raw_block", "x{% raw %}{{ not_run }} {% if %}{% endraw %}y", {}),
    ("trailing_newline_dropped", "ends with newline\n", {}),
    ("crlf_source", "a\r\nb\r\n{% if x %}c{% endif %}\r\n", {"x": True}),
    # values and output
    ("print_none", "{{ none }}|{{ missing }}|{{ true }}|{{ false }}", {}),
    ("print_numbers", "{{ 1.0 }} {{ 0.1 }} {{ 2.5e-7 }} {{ 1e21 }} {{ 7 // 2 }} {{ -7 // 2 }} {{ -7 % 3 }}", {}),
    ("arith", "{{ 2 ** 10 }} {{ 7 / 2 }} {{ 3 - 5 }} {{ -x }} {{ 1 + 2 * 3 }}", {"x": 4}),
    ("concat_tilde", "{{ 'a' ~ 1 ~ true ~ none ~ 2.0 }}", {}),
    ("list_repr", "{{ [1, 'two', none, true, [3.5]] }}", {}),
    ("dict_repr", "{{ {'b': 1, 'a': [2]} }}", {}),
    ("repr_quotes", "{{ ['it\\'s', \"say \\\"hi\\\"\", 'tab\\t'] }}", {}),
    ("compare_chain", "{{ 1 < 2 < 3 }} {{ 1 < 3 < 2 }} {{ 'a' < 'b' }} {{ 2 == 2.0 }} {{ true == 1 }}", {}),
    ("in_ops", "{{ 'b' in 'abc' }} {{ 3 not in [1, 2] }} {{ 'k' in {'k': 1} }} {{ 9 in [1] }}", {}),
    ("bool_logic", "{{ 0 or 'x' }} {{ 1 and 0 }} {{ not [] }} {{ none or false }}", {}),
    ("ternary", "{{ 'yes' if flag else 'no' }} {{ 'a' if missing_undefined is defined else 'b' }}", {"flag": 1}),
    ("slices", "{{ items[1:] }} {{ items[:-1] }} {{ items[::-1] }} {{ 'héllo'[1:3] }} {{ items[-1] }}",
     {"items": [1, 2, 3]}),
    ("index_missing", "{{ items[9] }}|{{ d['nope'] }}|{{ d.nope }}", {"items": [1], "d": {"a": 1}}),
    ("string_methods", "{{ ' x '.strip() }}|{{ 'a,b,,c'.split(',') }}|{{ 'abc'.startswith(('x', 'a')) }}"
     "|{{ 'Hello'.lower() }}|{{ 'a-b'.replace('-', '+') }}|{{ 'banana'.count('an') }}", {}),
    ("split_whitespace", "{{ '  a  b\tc\n'.split() }}|{{ 'a b c'.split(None, 1) }}", {}),
    ("dict_methods", "{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}{{ d.keys() | list }}"
     "{{ d.get('x', 'dflt') }}", {"d": {"b": 2, "a": 1}}),
    ("dict_method_shadow", "{{ d.items is callable }} {{ d['items'] }}", {"d": {"items": 5}}),
    # filters
    ("filter_trim", "[{{ '  a \n' | trim }}]", {}),
    ("filter_tojson", "{{ value | tojson }}", {"value": {"b": [1, 2.5, None, True], "a": "é\"x\""}}),
    ("filter_tojson_indent", "{{ value | tojson(indent=2) }}", {"value": {"a": [1, {"b": 2}], "c": []}}),
    ("filter_tojson_sort", "{{ value | tojson(sort_keys=true) }}", {"value": {"b": 1, "a": {"d": 2, "c": 3}}}),
    ("filter_tojson_ascii", "{{ value | tojson(ensure_ascii=true) }}", {"value": "é 😀"}),
    ("filter_dictsort", "{% for k, v in d | dictsort %}{{ k }}:{{ v }} {% endfor %}", {"d": {"b": 1, "B": 2, "a": 3}}),
    ("filter_dictsort_value", "{{ d | dictsort(false, 'value') }}", {"d": {"x": 2, "y": 1}}),
    ("filter_length", "{{ items | length }} {{ 'héllo' | length }} {{ d | length }}", {"items": [1, 2], "d": {}}),
    ("filter_join", "{{ items | join(', ') }}|{{ items | join }}", {"items": ["a", "b", "c"]}),
    ("filter_default", "{{ missing | default('dflt') }}|{{ '' | default('x', true) }}|{{ 0 | default(5) }}", {}),
    ("filter_first_last", "{{ items | first }} {{ items | last }} {{ 'abc' | first }}", {"items": [7, 8, 9]}),
    ("filter_upper_capitalize", "{{ 'hello world' | upper }} {{ 'hello' | capitalize }} {{ 'hello world' | title }}", {}),
    ("filter_int_float", "{{ '42' | int }} {{ 'x' | int(9) }} {{ '2.5' | float }} {{ 3.9 | int }}", {}),
    ("filter_string_list", "{{ 123 | string }} {{ 'ab' | list }} {{ d | list }}", {"d": {"k": 1}}),
    ("filter_reverse_sort", "{{ items | reverse | list }} {{ items | sort | list }} {{ 'abc' | reverse }}",
     {"items": [3, 1, 2]}),
    ("filter_sum_min_max", "{{ items | sum }} {{ items | min }} {{ items | max }}", {"items": [4, 1, 9]}),
    ("filter_select_reject", "{{ items | select | list }}|{{ items | reject('odd') | list }}|"
     "{{ items | select('gt', 2) | list }}", {"items": [1, 2, 3, 4]}),
    ("filter_selectattr_map", "{{ users | selectattr('active') | map(attribute='name') | join(',') }}",
     {"users": [{"name": "a", "active": True}, {"name": "b", "active": False}, {"name": "c", "active": True}]}),
    ("filter_unique_round_abs", "{{ items | unique | list }} {{ 2.567 | round(2) }} {{ -3 | abs }}", {"items": [1, 1, 2]}),
    ("filter_escape_safe", "{{ '<a & \"b\">' | e }}|{{ '<b>' | safe }}", {}),
    ("filter_replace", "{{ 'a-b-c' | replace('-', '') }} {{ 'aaa' | replace('a', 'b', 2) }}", {}),
    ("unknown_filter", "{{ 1 | nosuchfilter }}", {}),
    # tests
    ("tests_basic", "{{ x is defined }} {{ y is defined }} {{ none is none }} {{ 'a' is string }} {{ 1 is number }} "
     "{{ true is integer }} {{ 2 is odd }} {{ 4 is even }} {{ 6 is divisibleby 3 }}", {"x": 1}),
    ("tests_iterable_mapping", "{{ [] is iterable }} {{ {} is mapping }} {{ 'x' is sequence }} {{ 1.5 is float }}", {}),
    ("tests_compare", "{{ 3 is eq 3 }} {{ 3 is ne 4 }} {{ 3 is gt 2 }} {{ 3 is lt 2 }} {{ 'a' is in ['a'] }}", {}),
    ("test_not", "{{ x is not none }} {{ x is not defined }}", {"x": 0}),
    ("unknown_test", "{{ 1 is nosuchtest }}", {}),
    # loops
    ("loop_vars", "{% for x in items %}{{ loop.index }}/{{ loop.index0 }}{% if loop.first %}F{% endif %}"
     "{% if loop.last %}L{% endif %}/{{ loop.length }}/{{ loop.revindex }} {% endfor %}", {"items": ["a", "b", "c"]}),
    ("loop_else", "{% for x in [] %}x{% else %}empty{% endfor %}", {}),
    ("loop_filter", "{% for x in items if x % 2 == 0 %}{{ x }}{{ loop.length }} {% endfor %}",
     {"items": [1, 2, 3, 4]}),
    ("loop_break_continue", "{% for x in items %}{% if x == 2 %}{% continue %}{% endif %}{% if x == 4 %}"
     "{% break %}{% endif %}{{ x }}{% endfor %}", {"items": [1, 2, 3, 4, 5]}),
    ("loop_unpack_dict", "{% for k, v in d.items() %}{{ k }}{{ v }}{% endfor %}", {"d": {"a": 1, "b": 2}}),
    ("loop_cycle", "{% for x in items %}{{ loop.cycle('odd', 'even') }}{% endfor %}", {"items": [1, 2, 3]}),
    ("loop_over_string", "{% for c in 'hé' %}[{{ c }}]{% endfor %}", {}),
    ("loop_scope", "{% set x = 1 %}{% for i in [1] %}{% set x = 2 %}{% endfor %}{{ x }}", {}),
    ("namespace_mutate", "{% set ns = namespace(n=0) %}{% for i in items %}{% set ns.n = ns.n + i %}{% endfor %}"
     "{{ ns.n }}", {"items": [1, 2, 3]}),
    ("range_fn", "{% for i in range(3) %}{{ i }}{% endfor %}{{ range(1, 9, 3) | list }}", {}),
    # macros
    ("macro_default", "{% macro m(a, b='dflt') %}<{{ a }}|{{ b }}>{% endmacro %}{{ m(1) }}{{ m(2, b='x') }}", {}),
    ("macro_recursive", "{% macro f(n) %}{% if n > 0 %}{{ n }}{{ f(n - 1) }}{% endif %}{% endmacro %}{{ f(3) }}", {}),
    ("set_block", "{% set body %}inner {{ v }}{% endset %}[{{ body }}]", {"v": 7}),
    ("filter_block", "{% filter upper %}shout {{ v }}{% endfilter %}", {"v": "x"}),
    ("generation_block", "a{% generation %}b{{ v }}{% endgeneration %}c", {"v": 1}),
    # errors
    ("raise_exception", "{{ raise_exception('bad input') }}", {}),
    ("syntax_error_unclosed", "{% if x %}open", {"x": True}),
    ("undefined_attribute", "{{ missing.attr }}", {}),
    ("not_callable", "{{ 'abc'() }}", {}),
    # the chat template's own idioms
    ("messages_loop", "{%- for m in messages -%}{{ m['role'] }}:{{ m.content | trim }}{% if not loop.last %}|{% endif %}"
     "{%- endfor -%}", {"messages": [{"role": "user", "content": " hi "}, {"role": "assistant", "content": "yo"}]}),
    ("nested_tojson_args", "{%- for k, v in args | dictsort(true) -%}{{ k }}={{ v if v is string else v | tojson }};"
     "{%- endfor -%}", {"args": {"z": [1, "a"], "a": {"q": None}, "m": "text"}}),
]


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env()
    count = 0
    with open(out_dir / "jinja.jsonl", "w", encoding="utf-8") as out:
        for name, template, variables in CASES:
            row = {"name": name, "template": template, "vars": variables}
            try:
                row["expected"] = env.from_string(template).render(**variables)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            out.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
            count += 1
    print(json.dumps({"cases": count}))


if __name__ == "__main__":
    main()
