#!/usr/bin/env bash
# SDAR-8B on GSM8K in LMDeploy's PyTorch engine with its real KV-cache quantization, one card,
# the quant policies one after another (venv ~/venvs/lmdeploy, see results/stage1_report.md).
#   CARD=1 BLOCK=16 N=20 POLICIES="0 8 4 42" bash scripts/lmdeploy_sdar.sh    # smoke test
#   CARD=1 BLOCK=16      POLICIES="0 8 4 42" bash scripts/lmdeploy_sdar.sh    # all 1319
# Policies: 0 none, 4 int4, 8 int8, 16 fp8 e4m3, 17 fp8 e5m2, 42 TurboQuant (K4 V2).
# Output: out/lmdeploy/<N|full>/sdar8b_b<B>_t<T>_q<P>.jsonl (+ .meta.json, .log); reruns resume.
# A policy that fails is logged and the queue goes on.
set -uo pipefail
cd "$(dirname "$0")/.."
BLOCK=${BLOCK:-16}
T=${THRESHOLD:-0.9}
N=${N:-}
BATCH=${BATCH:-1}
REPO=JetLM/SDAR-8B-Chat
[[ $BLOCK != 4 ]] && REPO+="-b$BLOCK"
OUT=out/lmdeploy/${N:-full}
[[ $BATCH != 1 ]] && OUT+="_bs$BATCH"
mkdir -p "$OUT"
for P in ${POLICIES:-0 8}; do
  stem=sdar8b_b${BLOCK}_t${T}_q${P}
  echo "$(date '+%F %T') start $stem on card $CARD" | tee -a "$OUT/queue.log"
  CUDA_VISIBLE_DEVICES=$CARD HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  HF_DATASETS_CACHE=${HF_DATASETS_CACHE_OWN:-/data/caches/evgeniystep/hf_datasets} \
  python -m blockkv.lmdeploy_gsm8k --model-path "${MODEL_PATH:-$REPO}" --block "$BLOCK" --threshold "$T" \
    --quant-policy "$P" --batch "$BATCH" ${N:+--limit $N} --out "$OUT/$stem.jsonl" > "$OUT/$stem.log" 2>&1
  echo "$(date '+%F %T') end $stem rc=$?" | tee -a "$OUT/queue.log"
done
