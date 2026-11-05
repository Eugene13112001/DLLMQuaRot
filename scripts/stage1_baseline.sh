#!/usr/bin/env bash
# Stage 1.1: uncompressed baselines on GSM8K.
#   N=20  bash scripts/stage1_baseline.sh    # smoke test: read the answers by eye
#   N=200 bash scripts/stage1_baseline.sh
#         bash scripts/stage1_baseline.sh    # all 1319
# Env: MODELS_DIR (checkpoints named as on HF, e.g. $MODELS_DIR/SDAR-8B-Chat-b16),
#      GSM8K (test.jsonl/.parquet; default HF openai/gsm8k), GPUS (default "0"),
#      RUNS (model:block[:threshold], default below), REPEAT=1 to rerun the first
#      configuration into *_rep.jsonl for the determinism check.
# Threshold 1.0 (the default) is the authors' evaluation, for checking against their
# numbers (and, for Fast-dLLM v2, with SoloQ's); the 0.9 runs are the reference
# every compressed run is compared with.
set -euo pipefail
cd "$(dirname "$0")/.."

N=${N:-}
GPUS=(${GPUS:-0})
RUNS=${RUNS:-"sdar8b:4 sdar8b:8 sdar8b:16 sdar8b:32 sdar8b:64
  sdar8b:4:0.9 sdar8b:8:0.9 sdar8b:16:0.9 sdar8b:32:0.9 sdar8b:64:0.9
  fastdllm:32 fastdllm:32:0.9"}
OUT=out/stage1/${N:-full}
mkdir -p "$OUT"

common=()
[[ -n "$N" ]] && common+=(--limit "$N")
[[ -n "${GSM8K:-}" ]] && common+=(--data "$GSM8K")

name() {  # model:block[:threshold] -> file stem
  IFS=: read -r m b t <<< "$1"
  echo "${m}_b${b}${t:+_t$t}"
}

run() {  # spec gpu suffix
  local m b t extra=() stem
  IFS=: read -r m b t <<< "$1"
  stem=$(name "$1")$3
  [[ -n "$t" ]] && extra+=(--threshold "$t")
  [[ $m == fastdllm ]] && extra+=(--check-native 3)
  CUDA_VISIBLE_DEVICES=$2 python -m blockkv.run_gsm8k --model "$m" --block "$b" \
    --out "$OUT/$stem.jsonl" ${common[@]+"${common[@]}"} ${extra[@]+"${extra[@]}"} \
    > "$OUT/$stem.log" 2>&1 || echo "FAILED: $1 (see $OUT/$stem.log)"
}

# One configuration per GPU at a time; configurations queue round-robin.
i=0
for gpu in "${GPUS[@]}"; do
  (
    j=0
    for r in $RUNS; do
      if (( j % ${#GPUS[@]} == i )); then run "$r" "$gpu" ""; fi
      j=$((j + 1))
    done
  ) &
  i=$((i + 1))
done
wait

if [[ "${REPEAT:-0}" == 1 ]]; then
  first=$(echo $RUNS | cut -d' ' -f1)
  run "$first" "${GPUS[0]}" _rep
  python -m blockkv.summarize --compare "$OUT/$(name "$first").jsonl" "$OUT/$(name "$first")_rep.jsonl"
fi

files=()
for r in $RUNS; do
  f="$OUT/$(name "$r").jsonl"
  [[ -s "$f" ]] && files+=("$f")
done
if (( ${#files[@]} )); then
  python -m blockkv.summarize "${files[@]}"
else
  echo "No finished configurations: see $OUT/*.log"
fi
