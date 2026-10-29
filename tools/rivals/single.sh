#!/bin/bash
# Single-stream and CPU cells: llama.cpp's llama-bench and Dromos's `dromos bench` on the same weights.
#
# Rival cells (llama.cpp) and Dromos cells are separate modes, so each lock hold is one engine's cells:
#
#   llama-gpu      llama-bench, RTX 5070 Ti, full offload              (scripts/remote.sh rivals x ...)
#   llama-gpu-bf16 llama-bench, BF16 GGUF on the card (Praxis: the matched format for Dromos BF16)
#   llama-cpu      llama-bench, no GPU, 14 threads                     (scripts/remote.sh rivals run ...)
#   llama-offload  llama-bench Logos BF16 at NGL layers on the GPU     (scripts/remote.sh rivals x ...)
#   llama-experts  llama-bench Logos BF16, routed experts on the CPU    (scripts/remote.sh rivals x ...)
#   dromos-gpu / dromos-cpu / dromos-offload   the Dromos side, run once on final main
#
# llama-bench runs at -p 512 -n 128,256 (pp512, tg128, tg256). `dromos bench` runs one stream of 512 prompt
# tokens and 256 generated (greedy, synthetic prompts, the engine's own harness); its decode tok/s is
# "decode" in its report. Results go to out/rivals/single-*.log and out/rivals/single-*.json.
#
# LLAMA is the directory of llama.cpp's binaries (built at the pinned commit; see docs/RIVALS.md).
# NGL lists the llama.cpp -ngl values of the offload cells (the largest that fits is the headline).
set -euo pipefail

mode=${1:?gpu, cpu or offload}
here=$(cd "$(dirname "$0")/../.." && pwd)
cd "$here"
out=$here/out/rivals
mkdir -p "$out"

LLAMA=${LLAMA:-$HOME/lernex-engine-dev/rivals-llama/build-rivals/bin}
ENG=$HOME/kronos-scratch/logos/engines
PRAXIS_GGUF=$ENG/praxis/release
LOGOS_GGUF=$ENG/gguf/release
PRAXIS_BF16=$HOME/.cache/dromos/models/Lernex/Metis-1.6-Praxis
PRAXIS_FP8=$ENG/praxis/fp8
PRAXIS_NVFP4=$ENG/praxis/nvfp4
LOGOS_BF16=$HOME/.cache/dromos/models/Lernex/Metis-1.6-Logos
LOGOS_FP8=$HOME/.cache/dromos/models/Lernex/Metis-1.6-Logos-FP8

# Dromos reads its backend from LX_KERNELS first: clear it so LX_BACKEND=cpu selects the CPU library.
dromos=build/dromos

llama_bench() {  # name gguf [extra llama-bench args...]
  local name=$1 gguf=$2; shift 2
  if ! "$LLAMA/llama-bench" -m "$gguf" -p 512 -n 128,256 -r 3 -o json "$@" >"$out/single-$name.json" 2>"$out/single-$name.log"; then
    echo "llama-bench $name: did not run (see $out/single-$name.log: $(tail -1 "$out/single-$name.log" | cut -c1-120))"
    return 0
  fi
  echo "llama-bench $name: $("${PY:-python3}" -c 'import json,sys; print(" ".join(("pp%d=%.1f" % (r["n_prompt"], r["avg_ts"])) if r["n_prompt"] else ("tg%d=%.1f" % (r["n_gen"], r["avg_ts"])) for r in json.load(open(sys.argv[1]))))' "$out/single-$name.json")"
}

lx_bench() {  # name dir [extra dromos bench args...]
  local name=$1 dir=$2; shift 2
  $dromos bench "$dir" --streams 1 --prompt-tokens 512 --gen-tokens "${GEN:-256}" "$@" >"$out/single-$name.log" 2>&1
  echo "dromos bench $name:"; grep -E "decode|prefill|plan|Plan|weights|format" "$out/single-$name.log" | head -12
}

case $mode in
  llama-gpu)
    llama_bench praxis-q4km "$PRAXIS_GGUF/Metis-1.6-Praxis-Q4_K_M.gguf" -ngl 99
    llama_bench praxis-q8_0 "$PRAXIS_GGUF/Metis-1.6-Praxis-Q8_0.gguf" -ngl 99
    llama_bench logos-q4km "$LOGOS_GGUF/Metis-1.6-Logos-Q4_K_M.gguf" -ngl 99
    llama_bench logos-q8_0 "$LOGOS_GGUF/Metis-1.6-Logos-Q8_0.gguf" -ngl 99
    ;;
  llama-gpu-bf16)
    llama_bench praxis-bf16 "$PRAXIS_GGUF/Metis-1.6-Praxis-BF16.gguf" -ngl 99
    ;;
  llama-cpu)
    export CUDA_VISIBLE_DEVICES=
    llama_bench praxis-q4km-cpu "$PRAXIS_GGUF/Metis-1.6-Praxis-Q4_K_M.gguf" -ngl 0 -t 14
    llama_bench praxis-q8_0-cpu "$PRAXIS_GGUF/Metis-1.6-Praxis-Q8_0.gguf" -ngl 0 -t 14
    llama_bench logos-q4km-cpu "$LOGOS_GGUF/Metis-1.6-Logos-Q4_K_M.gguf" -ngl 0 -t 14
    llama_bench logos-q8_0-cpu "$LOGOS_GGUF/Metis-1.6-Logos-Q8_0.gguf" -ngl 0 -t 14
    ;;
  llama-offload)
    # NGL is a list: each value is one llama-bench run of the same BF16 file (a fit check first, then the numbers).
    for ngl in ${NGL:?set NGL, e.g. "20 22 24"}; do
      llama_bench logos-bf16-ngl$ngl "$LOGOS_GGUF/Metis-1.6-Logos-BF16.gguf" -ngl "$ngl" -p 512 -n 256 -r 2
    done
    ;;
  llama-experts)
    # The routed experts (ffn_*_exps, 18 GB of the 20.5 GB file) stay in host memory and the rest runs on the card:
    # the MoE offload Dromos makes on its own, expressed with llama.cpp's tensor override.
    # At the default micro-batch (512) the prompt-processing compute buffer alone is ~18 GB on this build, so the
    # prefill runs in smaller micro-batches (-ub); decode is one token per step and does not change.
    for ub in ${UB:-128 64}; do
      llama_bench logos-bf16-experts-cpu-ub$ub "$LOGOS_GGUF/Metis-1.6-Logos-BF16.gguf" -ngl 99 -ot "ffn_.*_exps=CPU" -ub "$ub" -p 512 -n 128,256 -r 2
    done
    ;;
  dromos-gpu)
    lx_bench praxis-nvfp4 "$PRAXIS_NVFP4"
    lx_bench praxis-fp8 "$PRAXIS_FP8"
    lx_bench praxis-bf16 "$PRAXIS_BF16"
    lx_bench logos-fp8 "$LOGOS_FP8"
    ;;
  dromos-cpu)
    export CUDA_VISIBLE_DEVICES=
    export LX_BACKEND=cpu LX_THREADS=14
    unset LX_KERNELS
    GEN=128 lx_bench praxis-nvfp4-cpu "$PRAXIS_NVFP4"
    GEN=128 lx_bench praxis-int8-cpu "$PRAXIS_BF16" --weights int8
    GEN=128 lx_bench logos-fp8-cpu "$LOGOS_FP8"
    ;;
  dromos-offload)
    lx_bench logos-bf16-offload "$LOGOS_BF16"
    ;;
  *) echo "mode must be llama-gpu, llama-cpu, llama-offload, llama-experts, dromos-gpu, dromos-cpu or dromos-offload" >&2; exit 2 ;;
esac
