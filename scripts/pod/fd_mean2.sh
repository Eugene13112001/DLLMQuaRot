# The two cells missing from the SageAttention comparison.
#
# Every cell of the bias table was taken with the exact removal of RoPE(b), and the mean
# matched it or beat it wherever both were run (83.0 against 84.0 per token at 4 bits,
# 79.5 against 80.5 per channel at 3 bits, 83.0 against 79.0 on QuaRot). The per-channel
# cell at 4 bits -- the one the paper's table leads with, 74.0 -> 85.0 -- has no mean twin,
# so the table cannot be stated in the form we recommend. This fills it, and adds the two-bit
# per-channel cell, where the exact form recovered only part of the loss (18.0 -> 43.5) and
# a data-driven vector may do better or worse.
#
#   tmux new -s fdm -d "bash scripts/pod/fd_mean2.sh 2>&1 | tee out/_fdm.log"

. "$(dirname "$0")/lib.sh"

MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B"
EVALFD="bash scripts/llada2.sh scripts/evaluate_fdv2.py $MFD"
C="--n-eval 200 --gen-length 2048 --threshold 0.95 --block-size 32 --small-block-size 8"
e() { n=$1; shift; run "$n" 34000 "out/$n.json" $EVALFD $C "$@" --out "out/$n.json"; }

e fd_4_ch_mean --kv-bits 4 --key-mean
e fd_2_ch_mean --kv-bits 2 --key-mean

echo "$(date +%H:%M) недостающие клетки со средним по SageAttention — готово"
