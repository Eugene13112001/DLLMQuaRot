#!/usr/bin/env bash
# SDAR on RULER (long inputs): does the harm of KV quantization depend on the block?
#   DATA=~/data/ruler GPUS="0 1" bash scripts/ruler.sh
# Env: DATA (required), RUNS (block:bits, bits "-" = uncompressed; default blocks
#      4, 16, 64, uncompressed and 2 bit), LENGTHS (default "4096 16384"), GPUS
#      (one queue per entry, repeat a GPU for two queues), PER_TASK (30), LIMIT,
#      THRESHOLD (0.9), KV_WINDOW (0), STAGGER (90).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${DATA:?set DATA to the folder with the ruler parquet files}"
GPUS=(${GPUS:-0})
RUNS=${RUNS:-"4:- 16:- 64:- 4:2 16:2 64:2"}
LENGTHS=${LENGTHS:-"4096 16384"}
W=${KV_WINDOW:-0}

jobs_list=()
for len in $LENGTHS; do
  for r in $RUNS; do jobs_list+=("$len:$r"); done
done

run() {  # length:block:bits gpu
  local len b q stem
  IFS=: read -r len b q <<< "$1"
  stem="sdar8b_b${b}"
  local kv=()
  if [[ $q != - ]]; then
    stem+="_kv${q}"
    kv=(--kv-bits "$q" --kv-window "$W")
    [[ $W != 0 ]] && stem+="_w$W"
  fi
  local dir="out/ruler/$len"
  mkdir -p "$dir"
  local extra=()
  [[ -n "${LIMIT:-}" ]] && extra=(--limit "$LIMIT")
  CUDA_VISIBLE_DEVICES=$2 python -m blockkv.run_ruler --block "$b" --length "$len" --data "$DATA" \
    --per-task "${PER_TASK:-30}" --threshold "${THRESHOLD:-0.9}" --out "$dir/$stem.jsonl" "${kv[@]}" "${extra[@]}" \
    > "$dir/$stem.log" 2>&1 || echo "FAILED: $1 (see $dir/$stem.log)"
}

i=0
for gpu in "${GPUS[@]}"; do
  (
    sleep $((i * ${STAGGER:-90}))
    j=0
    for r in "${jobs_list[@]}"; do
      if (( j % ${#GPUS[@]} == i )); then run "$r" "$gpu"; fi
      j=$((j + 1))
    done
  ) &
  i=$((i + 1))
done
wait

for len in $LENGTHS; do python -m blockkv.summarize_ruler "out/ruler/$len" || true; done
