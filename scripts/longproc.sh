#!/usr/bin/env bash
# LongProc (long input, long structured output) with our decoders and fake KV quantization:
# the step tax and runaways on answers of up to 8K tokens.
#   LONGPROC=~/third_party/LongProc GPUS="0 2" MODEL=sdar8b RUNS="32:- 32:2 32:2k 32:2v" bash scripts/longproc.sh
# RUNS: block:spec, spec as in stage1_kv.sh: "-" uncompressed, 2/3/4 bits, 2k / 2v (keys / values
#   alone), k4v2 (own bits), optionally :rot (Hadamard, keys per token), :rotv (values only),
#   :kb0.0625 (Kitty-like key-channel boost to 8 bits), e.g. "32:2:kb0.0625".
# Env: LONGPROC (repo folder), MODEL (sdar8b | fastdllm), DATASETS (default the five 8K sets),
#   PER_TASK (50), MAX_INPUT (24576), GPUS (one queue per entry), LIMIT, THRESHOLD (0.9), LEVEL (8k:
#   output folder name).
# Output: out/longproc/<LEVEL>/<model>_b<B>[_kv<spec>].jsonl
# Summary: python -m blockkv.summarize_ruler out/longproc/8k; tax vs runaways:
#   python -m blockkv.tax_decompose --cap 9216 REF::RUN
set -euo pipefail
cd "$(dirname "$0")/.."
: "${LONGPROC:?set LONGPROC to the LongProc repository folder}"
GPUS=(${GPUS:-0})
MODEL=${MODEL:-sdar8b}
RUNS=(${RUNS:-32:- 32:2})
LEVEL=${LEVEL:-8k}
DS=${DATASETS:-path_traversal_8k html_to_tsv_8k tom_tracking_8k countdown_8k travel_planning_8k}
dir=out/longproc/$LEVEL
mkdir -p "$dir"

run() {  # block:spec[:variant] gpu
  local b q v stem
  IFS=: read -r b q v <<< "$1"
  stem="${MODEL}_b${b}"
  local kv=()
  if [[ $q != - ]]; then
    stem+="_kv$q"
    if [[ $q =~ ^k([0-9]+)v([0-9]+)$ ]]; then
      kv=(--kv-bits "${BASH_REMATCH[1]}" --kv-k-bits "${BASH_REMATCH[1]}" --kv-v-bits "${BASH_REMATCH[2]}")
    elif [[ $q == *[kv] ]]; then
      kv=(--kv-bits "${q%[kv]}" --kv-only "${q: -1}")
    else
      kv=(--kv-bits "$q")
    fi
    case "${v:-}" in
      rot) kv+=(--kv-rotate --key-mode token); stem+="_ktoken_rot" ;;
      rotv) kv+=(--kv-rotate-values); stem+="_rotv" ;;
      kb*) kv+=(--kv-key-boost "${v#kb}" --kv-key-boost-bits 8); stem+="_kb${v#kb}x8" ;;
      "") ;;
      *) echo "unknown variant $v"; return 1 ;;
    esac
  fi
  local extra=()
  [[ -n "${LIMIT:-}" ]] && extra+=(--limit "$LIMIT")
  CUDA_VISIBLE_DEVICES=$2 python -m blockkv.run_longproc --model "$MODEL" --block "$b" --longproc "$LONGPROC" \
    --datasets $DS --per-task "${PER_TASK:-50}" --max-input "${MAX_INPUT:-24576}" --threshold "${THRESHOLD:-0.9}" \
    --out "$dir/$stem.jsonl" "${kv[@]}" "${extra[@]}" > "$dir/$stem.log" 2>&1 || echo "FAILED: $1 (see $dir/$stem.log)"
}

i=0
for gpu in "${GPUS[@]}"; do
  ( j=0; for r in "${RUNS[@]}"; do (( j % ${#GPUS[@]} == i )) && run "$r" "$gpu"; j=$((j + 1)); done ) &
  i=$((i + 1)); sleep 30
done
wait
python -m blockkv.summarize_ruler "$dir" || true
