# Part 1: LLaDA-1.5 on the same grid and commit rule as the other two models.
#
# The comparison of the three models had LLaDA-1.5 at two bits only and on the fixed
# schedule, while LLaDA2.0-mini and Fast-dLLM-v2 are on the 0.95 confidence threshold at
# 4 / 3 / 2 bits. This puts LLaDA-1.5 on that grid: keys per channel, per token and QuaRot
# (rotation then per token), all under the threshold. The prefix is refreshed at block
# boundaries as in every earlier LLaDA-1.5 cell (--kv-rope post, as rotation requires).
#
#   tmux new -s p1a -d "bash scripts/pod/p1_l15.sh 2>&1 | tee out/_p1a.log"

. "$(dirname "$0")/lib.sh"

C="--n-eval 200 --gen-length 512 --eval-steps 256 --kv-cache --kv-rope post --threshold 0.95"
e() { n=$1; shift; run "$n" $NEED15 "out/$n.json" $EVAL15 $C "$@" --out "out/$n.json"; }

e l15_T16        --kv-bits 16
# two bits first: the width where the other two models separate the axes most
e l15_T2_ch      --kv-bits 2
e l15_T2_tok     --kv-bits 2 --kv-key-axis channel
e l15_T2_quarot  --kv-bits 2 --kv-key-axis channel --rotate-qk
e l15_T3_ch      --kv-bits 3
e l15_T3_tok     --kv-bits 3 --kv-key-axis channel
e l15_T3_quarot  --kv-bits 3 --kv-key-axis channel --rotate-qk
e l15_T4_tok     --kv-bits 4 --kv-key-axis channel
e l15_T4_quarot  --kv-bits 4 --kv-key-axis channel --rotate-qk
e l15_T4_ch      --kv-bits 4

echo "$(date +%H:%M) часть 1: LLaDA-1.5 на общей сетке — готово"
