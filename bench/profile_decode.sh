#!/bin/bash
# The decode-step profile (bench/profile_decode.py), on the GPU server, from this checkout:
#
#   scripts/remote.sh SLOT x 'bench/profile_decode.sh MODEL_DIR [OUT_DIR]'
#
# Two nsys runs of `dromos bench MODEL --streams 1`: one eager and serial (every kernel on one stream in issue order,
# so each kernel maps to the product it runs), one as the engine replays it (graph replay, for the step's span).
# Needs build/dromos (scripts/remote.sh SLOT build) and nsys. OUT_DIR (default out/prof) keeps the trace, the bench
# output and the table; the nsys reports are deleted once exported.
set -euo pipefail
model=${1:?model dir}
out=${2:-out/prof}
mkdir -p "$out"
rm -f "$out"/*.nsys-rep "$out"/*.sqlite

# The engine's own number first, without the profiler: 1 stream, the same work as the profile runs.
build/dromos bench "$model" --streams 1 --prompt-tokens 64 --gen-tokens 128 | tee "$out/bench.txt"

LX_NO_GRAPHS=1 LX_SERIAL_DECODE=1 LX_LINEAR_TRACE="$out/linear.txt" \
  nsys profile --cuda-graph-trace=node -o "$out/eager" --force-overwrite=true \
  build/dromos bench "$model" --streams 1 --prompt-tokens 64 --gen-tokens 16 > "$out/eager.log" 2>&1
nsys export --type sqlite -o "$out/eager.sqlite" --force-overwrite=true "$out/eager.nsys-rep" > /dev/null

nsys profile --cuda-graph-trace=node -o "$out/graph" --force-overwrite=true \
  build/dromos bench "$model" --streams 1 --prompt-tokens 64 --gen-tokens 128 > "$out/graph.log" 2>&1
nsys export --type sqlite -o "$out/graph.sqlite" --force-overwrite=true "$out/graph.nsys-rep" > /dev/null
rm -f "$out"/*.nsys-rep

"${PY:-python3}" bench/profile_decode.py "$out/linear.txt" "$out/eager.sqlite" "$out/graph.sqlite" | tee "$out/table.txt"
