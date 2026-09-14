# Sourced by the queue scripts next to it. One job, one card, picked at start.
#
# The old queue picked a card once and ran every job on it. On a shared node
# the card that was emptiest at the start is not the emptiest an hour later,
# and two queues started together picked the *same* card, because neither
# model had landed yet when the other one looked. Both show up as
# "CUDA out of memory" while the model is being moved to the device.
#
# So here the card is picked per job, under a lock, and the lock is held until
# the model has had time to land -- the next pick then sees the memory as
# taken. A job that still dies of OOM is retried after a pause; one that dies
# of anything else is reported and left alone.

set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # same indices as nvidia-smi
export PYTHONUNBUFFERED=1
REPO="$HOME/quantization/DLLMQuaRot"
LOCK="/tmp/gpuq.$(id -un).lock"
SETTLE="${SETTLE:-240}"              # seconds the lock is held after a start
cd "$REPO" || exit 1
mkdir -p out

free_card() {  # $1 = MiB needed; prints the index of the emptiest card that has it
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', *' -v n="$1" '$2>n{print $2" "$1}' | sort -nr | head -1 | cut -d' ' -f2
}

# run NAME MIB OUTFILE cmd...   (cmd runs with CUDA_VISIBLE_DEVICES set)
run() {
  local n=$1 need=$2 out=$3; shift 3
  if [ -s "$out" ]; then echo "skip $n (есть $out)"; return 0; fi
  if pgrep -f -- "$out" >/dev/null; then echo "skip $n (уже идёт)"; return 0; fi
  local try c pid
  for try in 1 2 3; do
    exec 9>"$LOCK"; flock 9
    while c=$(free_card "$need"); [ -z "$c" ]; do
      echo "$(date +%H:%M) $n: нет карты с $need MiB, жду"; sleep 120
    done
    echo "$(date +%H:%M) === $n на GPU $c, попытка $try"
    # 9>&-: the job must not inherit the lock, or it would hold it for hours
    CUDA_VISIBLE_DEVICES=$c "$@" >"out/$n.log" 2>&1 9>&- &
    pid=$!
    sleep "$SETTLE"; flock -u 9; exec 9>&-
    wait "$pid"
    if [ -s "$out" ]; then echo "$(date +%H:%M) +++ $n готов"; return 0; fi
    if ! grep -q "OutOfMemoryError\|CUDA out of memory" "out/$n.log"; then
      echo "$(date +%H:%M) !!! $n упал не по памяти: tail out/$n.log"; return 1
    fi
    echo "$(date +%H:%M) $n: OOM, повтор через 5 минут"; sleep 300
  done
  echo "!!! $n: три OOM подряд, пропускаю"; return 1
}

EVAL15="env PYTHONPATH=$HOME/tf446 python scripts/evaluate.py --model GSAI-ML/LLaDA-1.5 --model-type llada"
EVAL20="bash scripts/llada2.sh scripts/evaluate.py --model inclusionAI/LLaDA2.0-mini --model-type llada2_moe"
PY15="env PYTHONPATH=$HOME/tf446 python"
PY20="bash scripts/llada2.sh"
M15="--model GSAI-ML/LLaDA-1.5 --model-type llada"
M20="--model inclusionAI/LLaDA2.0-mini --model-type llada2_moe"
NEED15=26000
NEED20=46000
