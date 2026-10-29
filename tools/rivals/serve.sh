#!/bin/bash
# One server, the same client, the same cells. Run it on the GPU server through the GPU lock:
#
#   scripts/remote.sh rivals x 'tools/rivals/serve.sh dromos NAME MODEL [server flags...]'
#   scripts/remote.sh rivals x 'tools/rivals/serve.sh sglang NAME MODEL [server flags...]'
#
# It starts the engine on MODEL, waits until it answers /v1/models, runs the cells with
# tools/rivals/load.py, stops the server (its whole process group, SGLang has children) and
# returns. Cells: the shared-prefix cells (64 and 128 streams sharing a 2,048-token prefix, 64
# unique tokens each, 256 generated) and the plain batched-decode cells (1, 8 and 32 streams of
# unique 512-token prompts, 256 generated).
#
#   NAME    the served model name and the label of the results: out/rivals/NAME-*.json
#   MODEL   a model directory; its tokenizer.json makes the prompts (the same bytes for both engines)
#
# Environment: PORT (default 8080 dromos, 30000 sglang), PREFIX_CACHE=0 turns the prefix cache off
# (dromos --no-prefix-cache, sglang --disable-radix-cache), CELLS=shared|unique|both|structured (default both;
# structured runs tools/rivals/structured.py alone),
# WAVES (measured waves per level, default 2), PY (client python), SGLANG_PY (sglang python).
set -euo pipefail

engine=${1:?engine: dromos or sglang}; shift
name=${1:?name}; shift
model=${1:?model directory}; shift
model=${model/#\~/$HOME}

here=$(cd "$(dirname "$0")/../.." && pwd)
cd "$here"
out=$here/out/rivals
mkdir -p "$out"
py=${PY:-$HOME/venvs/hfport/bin/python}
cells=${CELLS:-both}
waves=${WAVES:-2}
tok=$model/tokenizer.json
[ -f "$tok" ] || { echo "no tokenizer.json in $model" >&2; exit 2; }

nocache_on=${PREFIX_CACHE:-1}
case $engine in
  dromos)
    port=${PORT:-8080}
    nocache=(); [ "$nocache_on" = 0 ] && nocache=(--no-prefix-cache)
    cmd=(build/dromos serve "$model" --host 127.0.0.1 --port "$port" --name "$name" "${nocache[@]}" "$@")
    ;;
  sglang)
    port=${PORT:-30000}
    sglang_py=${SGLANG_PY:-$HOME/venvs/sglang/bin/python}
    nocache=(); [ "$nocache_on" = 0 ] && nocache=(--disable-radix-cache)
    cmd=("$sglang_py" -m sglang.launch_server --model-path "$model" --served-model-name "$name"
         --host 127.0.0.1 --port "$port" --reasoning-parser metis --tool-call-parser metis
         "${nocache[@]}" "$@")
    ;;
  *) echo "engine must be dromos or sglang" >&2; exit 2 ;;
esac

log=$out/$name.server.log
echo "== starting: ${cmd[*]}" | tee "$log"
setsid "${cmd[@]}" >>"$log" 2>&1 &
pid=$!
stop() {
  kill -- -$pid 2>/dev/null || true
  wait $pid 2>/dev/null || true
  # SGLang's scheduler processes can outlive the launcher: wait until the card is free of them.
  for _ in $(seq 60); do
    pgrep -f "sglang.launch_server.*--port $port" >/dev/null 2>&1 || break
    sleep 2
  done
  pkill -f "sglang.*--port $port" 2>/dev/null || true
}
trap stop EXIT

# Both engines answer /v1/models once the weights are in and the graphs are captured.
deadline=$((SECONDS + 1500))
until curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; do
  kill -0 $pid 2>/dev/null || { echo "server exited; see $log" >&2; tail -40 "$log" >&2; exit 1; }
  [ $SECONDS -lt $deadline ] || { echo "server not ready after 25 minutes; see $log" >&2; exit 1; }
  sleep 5
done
echo "== ready on :$port" | tee -a "$log"

url=http://127.0.0.1:$port
if [ "$cells" = shared ] || [ "$cells" = both ]; then
  "$py" tools/rivals/load.py --url "$url" --model "$name" --tokenizer "$tok" \
    --shared-prefix 2048 --suffix 64 --gen 256 --concurrency 64,128 --waves "$waves" \
    --out "$out/$name-shared.json"
fi
if [ "$cells" = unique ] || [ "$cells" = both ]; then
  "$py" tools/rivals/load.py --url "$url" --model "$name" --tokenizer "$tok" \
    --shared-prefix 0 --prompt-tokens 512 --gen 256 --concurrency 1,8,32 --waves "$waves" \
    --out "$out/$name-unique.json"
fi
if [ "$cells" = structured ]; then
  "$py" tools/rivals/structured.py --url "$url" --model "$name" --tokenizer "$tok" \
    --concurrency 32 --requests 64 --gen 160 --out "$out/$name-structured.json"
fi
echo "== done $name"
