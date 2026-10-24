#!/bin/bash
# Train a lernex.block/1 drafter for a Metis target, end to end, in one command:
#
#   tools/draft/pipeline.sh praxis            # or logos, or a target checkpoint directory
#
# Run it from the engine checkout on the GPU server, after a build (build/lernex). Every stage is
# resumable and skipped once its output is complete, so the same command picks up where a stopped
# run left off:
#   1. prompts     tools/draft/prompts.py: SOURCE conversations (perfectblend downloads DSpark's
#                  open-perfectblend; local is an offline mix for smoke tests), N_TRAIN + N_EVAL
#   2. answers     tools/draft/regenerate.py (vLLM): the target answers every conversation
#   3. eval        lernex features over the eval set, kept on disk
#   4. train       lernex features streams the target's hidden states for EPOCHS epochs through a
#                  bounded queue in /dev/shm (QUEUE shards), and tools/draft/train_block.py trains
#                  on them as they come, deleting what it has read; the drafter lands in
#                  $WORK/train/draft (config.json, model.safetensors), what `lernex spec
#                  --drafters block --draft-head DIR` loads
# Every GPU stage holds the shared GPU lock (~/lernex-engine-dev/.gpu.lock) while it runs.
#
# Settings (environment): WORK, SOURCE (perfectblend), N_TRAIN (200000), N_EVAL (96), EPOCHS (3),
# MAX_LEN (2048 tokens per training sequence), MAX_ANSWER (2048), THINKING (0.2), BATCH (128),
# LR (3e-4), ANCHORS (512), QUEUE (3), SHARD_TOKENS (16384), LOCK (1; 0 runs without the lock),
# MEM_MAX (16G: each stage runs in a memory-capped systemd scope, so a runaway build or loader
# fails alone instead of running the shared server out of memory).
set -euo pipefail

name=${1:?usage: tools/draft/pipeline.sh praxis|logos|TARGET_DIR}
case "$name" in
  praxis) target=/home/aiadmin/kronos-scratch/logos/engines/praxis/hf ;;
  logos) target=/home/aiadmin/kronos-scratch/logos/hfport/fp8 ;;
  *) target=$name; name=$(basename "$target") ;;
esac
here=$(cd "$(dirname "$0")/../.." && pwd)
cd "$here"
WORK=${WORK:-$HOME/kronos-scratch/draft/$name}
SOURCE=${SOURCE:-perfectblend}
N_TRAIN=${N_TRAIN:-200000}
N_EVAL=${N_EVAL:-96}
EPOCHS=${EPOCHS:-3}
MAX_LEN=${MAX_LEN:-2048}
MAX_ANSWER=${MAX_ANSWER:-2048}
THINKING=${THINKING:-0.2}
BATCH=${BATCH:-128}
LR=${LR:-3e-4}
ANCHORS=${ANCHORS:-512}
QUEUE=${QUEUE:-3}
SHARD_TOKENS=${SHARD_TOKENS:-16384}
PY=${PY:-$HOME/venvs/hfport/bin/python}
VLLM_PY=${VLLM_PY:-$HOME/venvs/vllm-metis/bin/python}
queue_dir=/dev/shm/lx-draft-$name
mkdir -p "$WORK"
log() { echo "[$(date '+%F %T')] $*"; }
MEM_MAX=${MEM_MAX:-16G}
cap=()
command -v systemd-run > /dev/null && cap=(systemd-run --user --scope --quiet -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0)
gpu() {
  if [ "${LOCK:-1}" = 1 ]; then flock "$HOME/lernex-engine-dev/.gpu.lock" "${cap[@]}" "$@"; else "${cap[@]}" "$@"; fi
}
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

[ -x build/lernex ] || { echo "build/lernex is missing: build the engine first" >&2; exit 1; }
taps=$("$PY" -c "
import json, sys
sys.path.insert(0, 'tools/draft')
import block_drafter as bd
cfg = bd.default_config(json.load(open('$target/config.json')))
json.dump(cfg, open('$WORK/draft_config.json', 'w'), indent=1)
print(','.join(f'{p}:{l}' for p, l in cfg['taps']))")
log "target $target, work $WORK, taps $taps"

# 1. prompts
if [ ! -s "$WORK/train.prompts.jsonl" ]; then
  log "prompts from $SOURCE"
  "$PY" tools/draft/prompts.py "$WORK" --source "$SOURCE" --train "$N_TRAIN" --eval "$N_EVAL"
fi

# 2. the target's answers
for part in eval train; do
  if [ ! -f "$WORK/$part.jsonl.done" ]; then
    log "answers for $part ($(lines "$WORK/$part.prompts.jsonl") conversations)"
    gpu "$VLLM_PY" tools/draft/regenerate.py "$target" "$WORK/$part.prompts.jsonl" "$WORK/$part.jsonl" \
      --max-tokens "$MAX_ANSWER" --thinking "$THINKING"
  fi
done

# 3. eval features, once
if [ ! -f "$WORK/eval-shards/DONE" ]; then
  log "eval features"
  gpu env LX_NO_W8=1 build/lernex features "$target" "$WORK/eval.jsonl" "$WORK/eval-shards" --taps "$taps" \
    --queue 1000000 --max-len "$MAX_LEN" --shard-tokens "$SHARD_TOKENS"
fi

# 4. train: features and trainer side by side on the GPU, the queue between them
n_train=$(lines "$WORK/train.jsonl")
total=$((n_train * EPOCHS))
seen=0
[ -f "$WORK/train/position.json" ] && seen=$("$PY" -c "import json; print(json.load(open('$WORK/train/position.json'))['seen'])")
if [ "$seen" -ge "$total" ] && [ -f "$WORK/train/draft/model.safetensors" ]; then
  log "already trained: $WORK/train/draft"
  exit 0
fi
resume=()
[ "$seen" -gt 0 ] && resume=(--resume)
log "training on $n_train sequences x $EPOCHS epochs ($seen seen)"
train() {
  rm -rf "$queue_dir"
  mkdir -p "$queue_dir"
  # the producer marks the queue FAILED if it dies, so the trainer stops instead of waiting
  ( LX_NO_W8=1 build/lernex features "$target" "$WORK/train.jsonl" "$queue_dir" --taps "$taps" --epochs "$EPOCHS" \
      --queue "$QUEUE" --max-len "$MAX_LEN" --shard-tokens "$SHARD_TOKENS" --skip-seqs "$seen" \
      > "$WORK/features.log" 2>&1 || touch "$queue_dir/FAILED" ) &
  local producer=$!
  "$PY" tools/draft/train_block.py "$target" "$queue_dir" "$WORK/train" --config "$WORK/draft_config.json" \
    --total-seqs "$total" --batch-seqs "$BATCH" --lr "$LR" --anchors "$ANCHORS" --eval "$WORK/eval-shards" \
    "${resume[@]}" 2>&1 | tee -a "$WORK/train.log"
  local status=${PIPESTATUS[0]}
  [ "$status" -eq 0 ] || pkill -P "$producer" 2>/dev/null
  wait "$producer"
  rm -rf "$queue_dir"
  return "$status"
}
export -f train log
export target WORK taps EPOCHS QUEUE MAX_LEN SHARD_TOKENS seen PY total BATCH LR ANCHORS queue_dir
gpu bash -c "$(declare -p resume); train"
log "drafter: $WORK/train/draft"
