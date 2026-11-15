#!/usr/bin/env bash
# BitSieve's own quality code (their prompt, their scoring, their default settings:
# block 32, sub-block 8, threshold 0.95, up to 512 new tokens). One config per card:
#   CARD=0 CFG=official_dense_bf16  bash scripts/bitsieve_quality.sh
#   CARD=1 CFG=dense_kivi2_k2v2_r32 bash scripts/bitsieve_quality.sh
#   CARD=1 CFG=official_dense_bf16 BENCH=qmsum bash scripts/bitsieve_quality.sh
# BENCH: gsm8k (default) or one of their LongBench tasks (qmsum: their own prompt, ROUGE-L);
# LongBench is read from LB_ZIP (default ~/data/longbench/data.zip) through their
# LONG_BENCH_ARCHIVE_PATH, no download. LIMIT: first N problems.
# Output: $BITSIEVE/results/check/$BENCH/$CFG.jsonl (+ .summary.json, .log). Re-running resumes.
# Compare: python -m blockkv.bitsieve_quality <official>.jsonl <kivi2>.jsonl
set -euo pipefail
BITSIEVE=${BITSIEVE:-$HOME/third_party/BitSieve}
BENCH=${BENCH:-gsm8k}
OUT=$BITSIEVE/results/check/$BENCH
# The shared /data/hf_cache/datasets is readable but its lock files belong to another user,
# so GSM8K is copied (a few MB) into our own datasets cache once.
DSC=${HF_DATASETS_CACHE_OWN:-/data/caches/evgeniystep/hf_datasets}
if [ ! -d "$DSC/openai___gsm8k" ]; then
  mkdir -p "$DSC" && cp -r /data/hf_cache/datasets/openai___gsm8k "$DSC/"
fi
mkdir -p "$OUT"
cd "$BITSIEVE"
CUDA_VISIBLE_DEVICES=$CARD HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_DATASETS_CACHE=$DSC PYTHONPATH=$BITSIEVE/src \
LONG_BENCH_ARCHIVE_PATH=${LB_ZIP:-$HOME/data/longbench/data.zip} \
nohup python -m bitsieve_fastdllm.eval.quality \
  --config "configs/$CFG.yaml" --benchmark "$BENCH" --split test ${LIMIT:+--limit $LIMIT} \
  --max-new-tokens 512 --max-cache-tokens 32768 \
  --output "$OUT/$CFG.jsonl" > "$OUT/$CFG.log" 2>&1 &
echo "started $CFG ($BENCH) on card $CARD, pid $!, log $OUT/$CFG.log"
