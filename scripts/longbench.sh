#!/usr/bin/env bash
# LongBench qmsum + gov_report (long input, long answer) with our decoders and fake KV quantization.
#   DATA=~/data/longbench/data.zip GPUS="3 3" MODEL=sdar8b RUNS="16:- 16:2" bash scripts/longbench.sh
#   DATA=... GPUS="3" MODEL=fastdllm RUNS="32:- 32:2" bash scripts/longbench.sh
# Env: DATA (data.zip or folder), MODEL (sdar8b | fastdllm), RUNS (block:bits, "-" uncompressed),
#      MAX_INPUT (prompt cut, default 16384), GPUS (one queue per entry), LIMIT, PER_TASK, THRESHOLD (0.9).
# For a separate QA series, set TASKS="2wikimqa narrativeqa musique" and OUT_DIR=out/longbench/16384_qa.
# Output: out/longbench/<MAX_INPUT>/<model>_b<B>[_kv<bits>].jsonl; summary: python -m blockkv.summarize_ruler DIR
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA:?set DATA to LongBench data.zip}"
GPUS=(${GPUS:-0})
MODEL=${MODEL:-sdar8b}
RUNS=(${RUNS:-16:- 16:2})
L=${MAX_INPUT:-16384}
dir=${OUT_DIR:-out/longbench/$L}
read -r -a tasks <<<"${TASKS:-qmsum gov_report}"
mkdir -p "$dir"
run() {  # block:bits gpu
  local b q stem
  IFS=: read -r b q <<< "$1"
  stem="${MODEL}_b${b}"
  local extra=()
  [[ $q != - ]] && { stem+="_kv$q"; extra+=(--kv-bits "$q"); }
  [[ -n "${LIMIT:-}" ]] && extra+=(--limit "$LIMIT")
  [[ -n "${PER_TASK:-}" ]] && extra+=(--per-task "$PER_TASK")
  CUDA_VISIBLE_DEVICES=$2 python -m blockkv.run_longbench --model "$MODEL" --block "$b" --data "$DATA" \
    --tasks "${tasks[@]}" --max-input "$L" --threshold "${THRESHOLD:-0.9}" --out "$dir/$stem.jsonl" "${extra[@]}" \
    > "$dir/$stem.log" 2>&1 || echo "FAILED: $1 (see $dir/$stem.log)"
}
i=0
for gpu in "${GPUS[@]}"; do
  ( j=0; for r in "${RUNS[@]}"; do (( j % ${#GPUS[@]} == i )) && run "$r" "$gpu"; j=$((j + 1)); done ) &
  i=$((i + 1)); sleep 30
done
wait
python -m blockkv.summarize_ruler "$dir" || true
