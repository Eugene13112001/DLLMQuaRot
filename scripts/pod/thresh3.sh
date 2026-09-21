# LLaDA2.0-mini cells that complete the axis comparison with Fast-dLLM-v2 at 3 and 4 bits.
#
#   tmux new -s thr3 -d "bash scripts/pod/thresh3.sh 2>&1 | tee out/_thr3.log"

. "$(dirname "$0")/lib.sh"

C="--n-eval 200 --gen-length 512 --eval-steps 256 --kv-cache --threshold 0.95"
e() { n=$1; shift; run "$n" $NEED20 "out/$n.json" $EVAL20 $C "$@" --out "out/$n.json"; }

e l20_T3_quarot --kv-bits 3 --kv-key-axis channel --rotate-qk
e l20_T4_tok    --kv-bits 4 --kv-key-axis channel
e l20_T4_quarot --kv-bits 4 --kv-key-axis channel --rotate-qk

echo "$(date +%H:%M) LLaDA2.0-mini, оси при 3 и 4 битах: готово"
