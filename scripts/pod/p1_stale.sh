# Part 1, thesis 5: under the confidence threshold, does staleness still leave the price of
# bits nearly additive when the current block is reused for longer?
#
# LLaDA2.0-mini under the threshold was measured at refresh every step and every 2 steps
# (price of 2 bits 9.0 -> 10.5). The fixed-schedule table showed its superadditivity mostly at
# every 4 (7.0 -> 31.0). This adds every 4 under the threshold.
#
#   tmux new -s p1c -d "bash scripts/pod/p1_stale.sh 2>&1 | tee out/_p1c.log"

. "$(dirname "$0")/lib.sh"

C="--n-eval 200 --gen-length 512 --eval-steps 256 --kv-cache --threshold 0.95"
e() { n=$1; shift; run "$n" $NEED20 "out/$n.json" $EVAL20 $C "$@" --out "out/$n.json"; }

e l20_T16_n4r --kv-bits 16 --reuse-window --kv-policy every_n --kv-refresh-every 4
e l20_T2_n4r  --kv-bits 2  --reuse-window --kv-policy every_n --kv-refresh-every 4

echo "$(date +%H:%M) часть 1: устаревание раз в 4 шага при пороге — готово"
