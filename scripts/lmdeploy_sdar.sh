#!/usr/bin/env bash
# SDAR-8B on GSM8K in LMDeploy's PyTorch engine with its real KV-cache quantization, one card,
# the quant policies one after another (venv ~/venvs/lmdeploy, see results/stage1_report.md).
#   BLOCK=16 N=20 POLICIES="0 8 4 42" bash scripts/lmdeploy_sdar.sh           # smoke test, waits for an empty card
#   CARD=1 BLOCK=16      POLICIES="0 8 4 42" bash scripts/lmdeploy_sdar.sh    # all 1319
# Policies: 0 none, 4 int4, 8 int8, 16 fp8 e4m3, 17 fp8 e5m2, 42 TurboQuant (K4 V2).
# Output: out/lmdeploy/<N|full>/sdar8b_b<B>_t<T>_q<P>.jsonl (+ .meta.json, .log); reruns resume.
# A policy that fails is logged and the queue goes on.
# BENCH=ruler LENGTH=4096|16384 RULER_DATA=~/data/ruler: RULER (the 390 problems of run_ruler) instead
# of GSM8K, output in out/lmdeploy/ruler<LENGTH>[_<N>][_bs<B>]/. CACHE_FRAC: share of free memory for
# the KV cache (default 0.3; raise it for big batches at 16K).
# BENCH=longbench LENGTH=16384 LB_DATA=~/data/longbench/data.zip: LongBench qmsum + gov_report (long input,
# answers up to 512 tokens), prompts cut to LENGTH, out/lmdeploy/longbench<LENGTH>[_<N>][_bs<B>]/.
# EXTRA: more arguments for blockkv.lmdeploy_gsm8k (e.g. "--eager --gen-length 512"), with TAG
# appended to the file stem (e.g. TAG=_eager_g512).
# Without CARD, each policy waits for an empty card among CARDS (default "0 1 2 3 4"):
# utilization <= 5% and >= NEED_GB (default 120) GiB free, 3 checks a minute apart
# (blockkv.wait_empty); the card and its state before and after go to queue.log.
set -uo pipefail
cd "$(dirname "$0")/.."
BLOCK=${BLOCK:-16}
T=${THRESHOLD:-0.9}
N=${N:-}
BATCH=${BATCH:-1}
REPO=JetLM/SDAR-8B-Chat
[[ $BLOCK != 4 ]] && REPO+="-b$BLOCK"
BENCH=${BENCH:-gsm8k}
if [[ $BENCH == longbench ]]; then
  OUT=out/lmdeploy/longbench${LENGTH:-16384}${N:+_$N}
  BARGS=(--bench longbench --length "${LENGTH:-16384}" --lb-data "${LB_DATA:-$HOME/data/longbench/data.zip}")
elif [[ $BENCH == ruler ]]; then
  OUT=out/lmdeploy/ruler${LENGTH:-4096}${N:+_$N}
  BARGS=(--bench ruler --length "${LENGTH:-4096}" --ruler-data "${RULER_DATA:-$HOME/data/ruler}")
else
  OUT=out/lmdeploy/${N:-full}
  BARGS=(--bench gsm8k)
fi
[[ $BATCH != 1 ]] && OUT+="_bs$BATCH"
mkdir -p "$OUT"
for P in ${POLICIES:-0 8}; do
  stem=sdar8b_b${BLOCK}_t${T}_q${P}${TAG:-}     # TAG: suffix for runs with EXTRA arguments
  card=${CARD:-}
  if [[ -z $card ]]; then
    echo "$(date '+%F %T') $stem: waiting for an empty card" >> "$OUT/queue.log"
    card=$(python -m blockkv.wait_empty --cards ${CARDS:-0 1 2 3 4} --need-gb "${NEED_GB:-120}" 2>> "$OUT/queue.log")
  fi
  state() { nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';'; }
  echo "$(date '+%F %T') start $stem on card $card | $(state)" | tee -a "$OUT/queue.log"
  CUDA_VISIBLE_DEVICES=$card HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  HF_DATASETS_CACHE=${HF_DATASETS_CACHE_OWN:-/data/caches/evgeniystep/hf_datasets} \
  python -m blockkv.lmdeploy_gsm8k --model-path "${MODEL_PATH:-$REPO}" --block "$BLOCK" --threshold "$T" \
    --quant-policy "$P" --batch "$BATCH" ${N:+--limit $N} "${BARGS[@]}" --cache-frac "${CACHE_FRAC:-0.3}" ${EXTRA:-} --out "$OUT/$stem.jsonl" > "$OUT/$stem.log" 2>&1
  rc=$?
  echo "$(date '+%F %T') end $stem rc=$rc | $(state)" | tee -a "$OUT/queue.log"
done
