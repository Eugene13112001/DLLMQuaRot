#!/usr/bin/env bash
# Wait for an idle GPU, then time the SDAR denoising step on it (blockkv.step_time).
#   nohup bash scripts/timing_when_idle.sh > out/timing/idle/watch.log 2>&1 &
# A card counts as idle when, CHECKS checks in a row: nvidia-smi shows utilization
# <= MAX_UTIL % and at most MAX_PROCS compute processes on it (matched to our CUDA index
# by UUID), and torch sees at least NEED_GB free. nvidia-smi here misses processes of
# other containers, so the free memory is the real check that nobody else is on the card.
# Env: CARDS (our CUDA indices to watch, default "0 1 2 3 4"),
#      MAX_UTIL (5), MAX_PROCS (0), NEED_GB (120 of 140), CHECKS (3), SLEEP (60 s between checks),
#      BLOCKS ("32 4 64"), LENGTHS ("1024 2048 4096 8192 16384"), BATCH ("1 4 16"),
#      ROUNDS (5), REPS (30), OUT (out/timing/idle).
set -uo pipefail
cd "$(dirname "$0")/.."

CARDS=${CARDS:-"0 1 2 3 4"}
MAX_UTIL=${MAX_UTIL:-5} MAX_PROCS=${MAX_PROCS:-0} NEED_GB=${NEED_GB:-120} CHECKS=${CHECKS:-3} SLEEP=${SLEEP:-60}
BLOCKS=${BLOCKS:-"32 4 64"} LENGTHS=${LENGTHS:-"1024 2048 4096 8192 16384"} BATCH=${BATCH:-"1 4 16"}
ROUNDS=${ROUNDS:-5} REPS=${REPS:-30} OUT=${OUT:-out/timing/idle}
mkdir -p "$OUT"

declare -A UUID COUNT
for c in $CARDS; do
  UUID[$c]=$(CUDA_VISIBLE_DEVICES=$c python -c "import torch; print(str(torch.cuda.get_device_properties(0).uuid)[:8])" 2>/dev/null)
  COUNT[$c]=0
  echo "card $c = GPU-${UUID[$c]}"
done

free_gb() { CUDA_VISIBLE_DEVICES=$1 python -c "import torch; print(int(torch.cuda.mem_get_info(0)[0] // 2**30))" 2>/dev/null || echo 0; }
util() { nvidia-smi --query-gpu=uuid,utilization.gpu --format=csv,noheader,nounits | grep "GPU-$1" | awk -F', ' '{print $2}'; }
procs() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "GPU-$1"; }

while :; do
  for c in $CARDS; do
    u=$(util "${UUID[$c]}"); u=${u:-100}
    p=$(procs "${UUID[$c]}"); p=${p:-99}
    f=-
    if (( u <= MAX_UTIL && p <= MAX_PROCS )); then
      f=$(free_gb "$c")
      if (( f >= NEED_GB )); then COUNT[$c]=$((COUNT[$c] + 1)); else COUNT[$c]=0; fi
    else
      COUNT[$c]=0
    fi
    echo "$(date '+%F %T') card $c util ${u}% procs ${p} free ${f} GiB idle-checks ${COUNT[$c]}/$CHECKS"
    if (( COUNT[$c] >= CHECKS )); then
      echo "card $c is idle: timing on it" | tee "$OUT/where.txt"
      for b in $BLOCKS; do
        CUDA_VISIBLE_DEVICES=$c HF_HUB_OFFLINE=1 python -m blockkv.step_time --block "$b" \
          --lengths $LENGTHS --batch $BATCH --rounds "$ROUNDS" --reps "$REPS" \
          --out "$OUT/sdar8b_b$b.jsonl" > "$OUT/sdar8b_b$b.log" 2>&1 || echo "FAILED: block $b (see $OUT/sdar8b_b$b.log)"
      done
      echo "done $(date '+%F %T')" >> "$OUT/where.txt"
      exit 0
    fi
  done
  sleep "$SLEEP"
done
