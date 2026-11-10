#!/usr/bin/env bash
# Stage 1.2: KV-cache quantization (KIVI-style, variant A) on GSM8K.
#   bash scripts/stage1_kv.sh                 # search: fixed random 400 problems
#   SUBSET= bash scripts/stage1_kv.sh         # final: all 1319
# Env: RUNS (model:block:threshold:bits[:key_mode], default: the 15 runs of stage 1.4;
#      bits 2/3/4/8/16, "2k" / "2v" quantize the keys / the values alone,
#      key_mode channel or token), GPUS (default "0";
#      repeat a GPU to run several configurations on it), SUBSET (default 400),
#      SEED (default 0), KV_GROUP (32), KV_WINDOW (0), KV_SINK (0), KEY_MODE
#      (channel), COMMIT_TEMP (1.0, DECAL's commit temperature), STAGGER (seconds between queue starts, 90), GSM8K, MODELS_DIR.
# Compare each run with its uncompressed reference on the same problems:
#   python -m blockkv.summarize --compare out/stage1/full/sdar8b_b16_t0.9.jsonl \
#                                         out/stage1_kv/400/sdar8b_b16_t0.9_kv2.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."

SUBSET=${SUBSET-400}
GPUS=(${GPUS:-0})
RUNS=${RUNS:-"sdar8b:4:0.9:2 sdar8b:8:0.9:2 sdar8b:16:0.9:2 sdar8b:32:0.9:2 sdar8b:64:0.9:2
  sdar8b:4:0.9:2:token sdar8b:8:0.9:2:token sdar8b:16:0.9:2:token sdar8b:32:0.9:2:token sdar8b:64:0.9:2:token
  sdar8b:4:0.9:4 sdar8b:8:0.9:4 sdar8b:16:0.9:4 sdar8b:32:0.9:4 sdar8b:64:0.9:4"}
G=${KV_GROUP:-32} W=${KV_WINDOW:-0} S=${KV_SINK:-0} KM=${KEY_MODE:-channel}
OUT=out/stage1_kv/${SUBSET:-full}
mkdir -p "$OUT"

CT=${COMMIT_TEMP:-1.0}
common=(--kv-group "$G" --kv-window "$W" --kv-sink "$S" --commit-temp "$CT")
[[ -n "$SUBSET" ]] && common+=(--subset "$SUBSET" --seed "${SEED:-0}")
[[ -n "${GSM8K:-}" ]] && common+=(--data "$GSM8K")

name() {  # model:block:threshold:bits[:key_mode] -> file stem; non-default settings go into the name
  local m b t q k
  IFS=: read -r m b t q k <<< "$1"
  k=${k:-$KM}
  local s="${m}_b${b}_t${t}_kv${q}"
  [[ $G != 32 ]] && s+="_g$G"
  [[ $W != 0 ]] && s+="_w$W"
  [[ $S != 0 ]] && s+="_s$S"
  [[ $k != channel ]] && s+="_k$k"
  [[ $CT != 1.0 ]] && s+="_T$CT"
  echo "$s"
}

run() {  # spec gpu
  local m b t q k stem
  IFS=: read -r m b t q k <<< "$1"
  k=${k:-$KM}
  stem=$(name "$1")
  local only=()
  [[ $q == *[kv] ]] && only=(--kv-only "${q: -1}")
  CUDA_VISIBLE_DEVICES=$2 python -m blockkv.run_gsm8k --model "$m" --block "$b" --threshold "$t" \
    --kv-bits "${q%[kv]}" "${only[@]}" --key-mode "$k" --out "$OUT/$stem.jsonl" "${common[@]}" \
    > "$OUT/$stem.log" 2>&1 || echo "FAILED: $1 (see $OUT/$stem.log)"
}

i=0
for gpu in "${GPUS[@]}"; do
  (
    sleep $((i * ${STAGGER:-90}))
    j=0
    for r in $RUNS; do
      if (( j % ${#GPUS[@]} == i )); then run "$r" "$gpu"; fi
      j=$((j + 1))
    done
  ) &
  i=$((i + 1))
done
wait

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
