#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print the waves of tools/rivals/load.py and structured.py result files as one table.

  python tools/rivals/summarize.py out/rivals/*-shared.json out/rivals/*-unique.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(paths: list[str]) -> int:
    print(f"{'file':40s} {'c':>4s} {'w':>2s} {'e2e tok/s':>10s} {'TTFT p50':>9s} {'TTFT p90':>9s} "
          f"{'stream':>7s} {'tokens':>9s} {'err':>4s}")
    for p in paths:
        data = json.loads(Path(p).read_text())
        name = Path(p).name
        for w in data.get("waves", []):
            if "concurrency" not in w:
                continue
            tok = f"{w.get('tokens_min', '-')}-{w.get('tokens_max', '-')}"
            print(f"{name:40s} {w['concurrency']:>4d} {w['wave']:>2d} {str(w.get('e2e_tok_s')):>10s} "
                  f"{str(w.get('ttft_p50_ms')):>9s} {str(w.get('ttft_p90_ms')):>9s} "
                  f"{str(w.get('stream_tok_s_median')):>7s} {tok:>9s} {w.get('error_count', 0):>4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
