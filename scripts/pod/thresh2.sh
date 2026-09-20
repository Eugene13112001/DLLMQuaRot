# The cells LLaDA2.0-mini is missing for a complete pair with Fast-dLLM-v2.
#
# Everything compared across the two models has to be decoded the same way, so the
# LLaDA2.0-mini side of the table is the threshold one (--threshold 0.95, as in the
# BitSieve harness and in Fast-dLLM-v2's own sampler), not the fixed-schedule numbers.
# thresh.sh took 16 bits, two bits on each axis and the gain migration; these are the
# four that were never run under the threshold: QuaRot, and the widths in between.
#
#   tmux new -s thr2 -d "bash scripts/pod/thresh2.sh 2>&1 | tee out/_thr2.log"

. "$(dirname "$0")/lib.sh"

C="--n-eval 200 --gen-length 512 --eval-steps 256 --kv-cache --threshold 0.95"
e() { n=$1; shift; run "$n" $NEED20 "out/$n.json" $EVAL20 $C "$@" --out "out/$n.json"; }

# the missing axis cell: rotation, the one Fast-dLLM-v2 will have
e l20_T2_quarot --kv-bits 2 --kv-key-axis channel --rotate-qk
# the widths in between, to put both models on the same width grid
e l20_T4_ch     --kv-bits 4
e l20_T3_ch     --kv-bits 3
e l20_T3_tok    --kv-bits 3 --kv-key-axis channel

echo "$(date +%H:%M) LLaDA2.0-mini при пороге: сетка достроена"
