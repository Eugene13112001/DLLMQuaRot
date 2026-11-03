#!/usr/bin/env bash
# Stage 1.1: uncompressed baselines on GSM8K.
#   N=20  bash scripts/stage1_baseline.sh    # smoke test: read the answers by eye
#   N=200 bash scripts/stage1_baseline.sh
#         bash scripts/stage1_baseline.sh    # all 1319
# Env: MODELS_DIR (checkpoints named as on HF, e.g. $MODELS_DIR/SDAR-8B-Chat-b16),
#      GSM8K (test.jsonl/.parquet; default HF openai/gsm8k), GPUS (default "0"),
#      RUNS (default: all five SDAR-8B blocks + Fast-dLLM v2), REPEAT=1 to rerun
#      the first configuration into *_rep.jsonl for the determinism check.
set -euo pipefail
cd "$(dirname "$0")/.."

N=${N:-}
GPUS=(${GPUS:-0})
RUNS=${RUNS:-"sdar8b:4 sdar8b:8 sdar8b:16 sdar8b:32 sdar8b:64 fastdllm:32"}
OUT=out/stage1/${N:-full}
mkdir -p "$OUT"

common=()
[[ -n "$N" ]] && common+=(--limit "$N")
[[ -n "${GSM8K:-}" ]] && common+=(--data "$GSM8K")

run() {  # model block gpu suffix
  local extra=()
  [[ $1 == fastdllm ]] && extra+=(--check-native 3)
  CUDA_VISIBLE_DEVICES=$3 python -m blockkv.run_gsm8k --model "$1" --block "$2" \
    --out "$OUT/$1_b$2$4.jsonl" "${common[@]}" "${extra[@]}" > "$OUT/$1_b$2$4.log" 2>&1
}

# One configuration per GPU at a time; configurations queue round-robin.
i=0
for gpu in "${GPUS[@]}"; do
  (
    j=0
    for r in $RUNS; do
      if (( j % ${#GPUS[@]} == i )); then run "${r%%:*}" "${r##*:}" "$gpu" ""; fi
      j=$((j + 1))
    done
  ) &
  i=$((i + 1))
done
wait

if [[ "${REPEAT:-0}" == 1 ]]; then
  first=${RUNS%% *}
  run "${first%%:*}" "${first##*:}" "${GPUS[0]}" _rep
  python -m blockkv.summarize --compare "$OUT/${first%%:*}_b${first##*:}.jsonl" "$OUT/${first%%:*}_b${first##*:}_rep.jsonl"
fi

python -m blockkv.summarize "$OUT"/*_b[0-9]*[0-9].jsonl
