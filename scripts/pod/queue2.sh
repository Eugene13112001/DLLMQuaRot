# Two-bit cache on task accuracy: does the way K is grouped start to matter where the tensor says it should?
#
# At 3-4 bits every scheme was free on both families. The key-error probe at 2 bits still has the per-channel
# scale well ahead of the per-token scale and of QuaRot, on both models, with the widest margin on LLaDA2.0-mini.
# These cells ask whether that reaches the answers. Only K changes between the first three cells of a model;
# V stays at its default (dynamic, one scale per token), so a difference is the key scheme and nothing else.
# The fourth cell is the practical recipe, static per-channel scales for K and V.
#
# LLaDA-1.5 stores K after RoPE in all four cells: R4 needs it, and the cells have to share a tensor.
# Idempotent like queue.sh: finished or running outputs are skipped.
#
#   tmux new -s b15 -d "bash scripts/pod/queue2.sh 15 2>&1 | tee out/_b15.log"
#   tmux new -s b20 -d "bash scripts/pod/queue2.sh 20 2>&1 | tee out/_b20.log"

M="${1:?model: 15|20}"
. "$(dirname "$0")/lib.sh"

C="--n-eval 200 --gen-length 512 --eval-steps 256 --kv-cache --kv-policy block --kv-bits 2"

if [ "$M" = 15 ]; then
  e() { n=$1; shift; run "$n" $NEED15 "out/$n.json" $EVAL15 $C --kv-rope post "$@" --out "out/$n.json"; }
  P=l15_B2
else
  e() { n=$1; shift; run "$n" $NEED20 "out/$n.json" $EVAL20 $C "$@" --out "out/$n.json"; }
  P=l20_B2
fi

e ${P}_ch                                          # K: one scale per channel (the recipe's axis)
e ${P}_tok     --kv-key-axis channel               # K: one scale per token
e ${P}_quarot  --kv-key-axis channel --rotate-qk   # K: R4, then one scale per token (QuaRot)
e ${P}_static  --kv-static kv                      # recipe: static per-channel scales for K and V

echo "$(date +%H:%M) очередь 2 бит, модель $M, прошла"
