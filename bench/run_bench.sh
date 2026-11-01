#!/bin/bash
# The measurement driver behind docs/BENCH.md. Runs on the GPU server, in the bench slot, through the GPU lock:
#
#   scripts/remote.sh bench x 'bash bench/run_bench.sh doctor MODEL TAG [format flags]'
#   scripts/remote.sh bench x 'bash bench/run_bench.sh engine MODEL TAG [format flags]'     dromos bench x3, prefill, spec-check
#   scripts/remote.sh bench x 'bash bench/run_bench.sh serve-engine MODEL TAG [format flags]'
#   scripts/remote.sh bench x 'bash bench/run_bench.sh serve-vllm MODEL TAG VOCAB [vllm flags]'
#   scripts/remote.sh bench x 'bash bench/run_bench.sh spec MODEL TAG [format flags]'
#   scripts/remote.sh bench x 'bash bench/run_bench.sh cachecheck MODEL TAG [format flags]'
#   scripts/remote.sh bench x 'bash bench/run_bench.sh cacheoff MODEL TAG [format flags]'
#   scripts/remote.sh bench x 'bash bench/run_bench.sh profile MODEL TAG STREAMS [format flags]'
#
# MODEL is a folder. TAG names the outputs. Format flags (--weights, --no-bf16x) go to dromos, as they are.
# Results go to $LX_BENCH_OUT (default ~/lernex-engine-dev/bench-results/2026-10-28), which is outside the slot
# tree, so the next scripts/remote.sh sync cannot delete them (rsync --delete only touches the slot).
#
# Every engine run disables the prefix cache (--no-prefix-cache), because the benchmark prompts repeat across
# runs and warm-ups: a warm cache would turn repeats into cache hits and inflate prefill and TTFT. vLLM runs
# with --no-enable-prefix-caching for the same reason.
set -u
cmd=${1:?command}; shift
OUT=${LX_BENCH_OUT:-$HOME/lernex-engine-dev/bench-results/2026-10-28}
mkdir -p "$OUT"
LX=${LX:-build/dromos}
PY=${PY:-$HOME/venvs/hfport/bin/python}
PROMPTS=${LX_PROMPTS:-${LX_WORK:-$HOME/lx-work}/logos/engines/work/praxisrefs/prompts.json}
ENGINE_PORT=${LX_ENGINE_PORT:-8811}
VLLM_PORT=${LX_VLLM_PORT:-8812}
VLLM_PY=${VLLM_PY:-$HOME/venvs/vllm-metis/bin/python}

log() { echo "[run_bench $(date -u +%H:%M:%S)] $*" >&2; }

# Waits until the server at $1 answers /v1/models, or the process $2 is gone, or $3 seconds pass.
wait_ready() {
  local url=$1 pid=$2 limit=$3 t=0
  while [ "$t" -lt "$limit" ]; do
    if curl -sf -m 5 "$url/v1/models" >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then log "the server exited before it was ready"; return 1; fi
    sleep 5; t=$((t + 5))
  done
  log "the server was not ready after $limit s"
  return 1
}

vocab_of() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1] + "/config.json"))["vocab_size"])' "$1"; }

case "$cmd" in
  doctor)
    model=$1; tag=$2; shift 2
    "$LX" doctor "$model" "$@" > "$OUT/$tag.doctor.txt" 2>&1
    ;;

  engine)
    model=$1; tag=$2; shift 2
    log "engine $tag: dromos bench x3"
    for rep in 1 2 3; do
      "$LX" bench "$model" --streams 1,8,32,64 --prompt-tokens 512 --gen-tokens 256 --no-prefix-cache "$@" \
        > "$OUT/$tag.bench.r$rep.txt" 2>&1
      log "bench rep $rep exit $?"
    done
    for T in 512 2048 8192; do
      "$LX" prefill "$model" "$PROMPTS" "$T" 5 "$@" > "$OUT/$tag.prefill.T$T.txt" 2>&1
      log "prefill T=$T exit $?"
    done
    ;;

  cachecheck)
    # The default dromos bench (prefix cache on) once, to measure what a warm cache does to prefill and TTFT.
    model=$1; tag=$2; shift 2
    "$LX" bench "$model" --streams 1,8,32,64 --prompt-tokens 512 --gen-tokens 256 "$@" \
      > "$OUT/$tag.bench-prefixcache.txt" 2>&1
    log "cachecheck $tag exit $?"
    ;;

  cacheoff)
    # The same run as cachecheck with the prefix cache off: the same prompts, one run, to compare against it.
    model=$1; tag=$2; shift 2
    "$LX" bench "$model" --streams 1,8,32,64 --prompt-tokens 512 --gen-tokens 256 --no-prefix-cache "$@" \
      > "$OUT/$tag.bench-nocache.txt" 2>&1
    log "cacheoff $tag exit $?"
    ;;

  spec)
    model=$1; tag=$2; shift 2
    "$LX" spec-check "$model" "$PROMPTS" "$@" > "$OUT/$tag.spec-check.txt" 2>&1
    log "spec-check $tag exit $?"
    ;;

  serve-engine)
    model=$1; tag=$2; shift 2
    url="http://127.0.0.1:$ENGINE_PORT"
    vocab=$(vocab_of "$model")
    log "serve-engine $tag on $ENGINE_PORT (vocab $vocab)"
    "$LX" serve "$model" --host 127.0.0.1 --port "$ENGINE_PORT" --name "$tag" --sequences 64 --context 9216 \
      --no-prefix-cache "$@" > "$OUT/$tag.serve.log" 2>&1 &
    pid=$!
    if wait_ready "$url" "$pid" 1800; then
      "$PY" bench/endpoint_bench.py --url "$url" --model "$tag" --vocab "$vocab" --streams 1,8,32,64 \
        --prompt-tokens 512 --gen-tokens 256 --repeats 3 --prefill 512,2048,8192 --prefill-runs 5 \
        --prompts "$PROMPTS" --label "$tag dromos-serve" --out "$OUT/$tag.serve.json" \
        > "$OUT/$tag.serve.txt" 2>&1
      log "client exit $?"
    fi
    kill -INT "$pid" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    ;;

  serve-vllm)
    model=$1; tag=$2; vocab=$3; shift 3
    url="http://127.0.0.1:$VLLM_PORT"
    unit="lx-bench-$tag-$(date +%s)"
    log "serve-vllm $tag on $VLLM_PORT, unit $unit"
    # One transient scope: its memory is capped (the server also runs CI runners), and stopping the scope stops the
    # engine core child with it. MAX_JOBS=2 keeps any JIT compile to two compilers.
    systemd-run --user --scope -q --unit="$unit" -p MemoryMax=20G \
      env MAX_JOBS=2 "$VLLM_PY" -m vllm.entrypoints.cli.main serve "$model" \
        --served-model-name "$tag" --host 127.0.0.1 --port "$VLLM_PORT" --trust-remote-code \
        --generation-config vllm --no-enable-prefix-caching --max-model-len 9216 \
        --max-num-seqs 64 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 "$@" \
        > "$OUT/$tag.vllm.log" 2>&1 &
    pid=$!
    if wait_ready "$url" "$pid" 2400; then
      # LX_VLLM_STREAMS narrows the stream counts; LX_VLLM_PREFILL narrows the prefill lengths (empty: none). A
      # context shorter than a prefill length cannot take that request, so set LX_VLLM_PREFILL to lengths that fit.
      "$PY" bench/endpoint_bench.py --url "$url" --model "$tag" --vocab "$vocab" \
        --streams "${LX_VLLM_STREAMS-1,8,32,64}" \
        --prompt-tokens 512 --gen-tokens 256 --repeats 3 --prefill "${LX_VLLM_PREFILL-512,2048,8192}" --prefill-runs 5 \
        --prompts "$PROMPTS" --metrics --label "$tag vllm" --out "$OUT/$tag.vllm.json" \
        > "$OUT/$tag.vllm.txt" 2>&1
      log "client exit $?"
    fi
    systemctl --user stop "$unit.scope" 2>/dev/null
    wait "$pid" 2>/dev/null
    ;;

  profile)
    # Kernel time of one dromos bench run at one stream count (CUDA graph nodes traced), for docs/BENCH.md's
    # "where the engine loses": nsys over the whole run, warm-up and prefill included, so read the decode kernels.
    model=$1; tag=$2; streams=$3; shift 3
    rep="$OUT/$tag.s$streams"
    nsys profile -t cuda --cuda-graph-trace=node --force-overwrite=true -o "$rep" \
      "$LX" bench "$model" --streams "$streams" --prompt-tokens 512 --gen-tokens 256 --no-prefix-cache "$@" \
      > "$rep.run.txt" 2>&1
    nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true "$rep.nsys-rep" > "$rep.kern.csv" 2>"$rep.stats.err"
    log "profile $tag s=$streams exit $?"
    ;;

  *)
    echo "unknown command $cmd" >&2
    exit 2
    ;;
esac
log "$cmd done"
