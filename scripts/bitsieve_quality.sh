#!/usr/bin/env bash
# BitSieve's own quality code on full GSM8K (their prompt, their scoring, their default settings:
# block 32, sub-block 8, threshold 0.95, up to 512 new tokens). One config per card:
#   CARD=0 CFG=official_dense_bf16  bash scripts/bitsieve_quality.sh
#   CARD=1 CFG=dense_kivi2_k2v2_r32 bash scripts/bitsieve_quality.sh
# Output: $BITSIEVE/results/check/gsm8k/$CFG.jsonl (+ .summary.json, .log). Re-running resumes.
# Compare: python -m blockkv.bitsieve_quality <official>.jsonl <kivi2>.jsonl
set -euo pipefail
BITSIEVE=${BITSIEVE:-$HOME/third_party/BitSieve}
OUT=$BITSIEVE/results/check/gsm8k
mkdir -p "$OUT"
cd "$BITSIEVE"
CUDA_VISIBLE_DEVICES=$CARD HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=$BITSIEVE/src \
nohup python -m bitsieve_fastdllm.eval.quality \
  --config "configs/$CFG.yaml" --benchmark gsm8k --split test \
  --max-new-tokens 512 --max-cache-tokens 32768 \
  --output "$OUT/$CFG.jsonl" > "$OUT/$CFG.log" 2>&1 &
echo "started $CFG on card $CARD, pid $!, log $OUT/$CFG.log"
