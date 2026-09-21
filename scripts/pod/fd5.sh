# Fast-dLLM-v2 against LLaDA2.0-mini on the key axis, at widths where both models survive.
#
# At two bits Fast-dLLM-v2 collapses on every axis (18 / 0 / 1), so the axis cannot be
# compared there: the MoE model shows an 88-point axis effect, this one a floor. At three
# bits LLaDA2.0-mini already separates the axes (93.0 per channel against 59.5 per token),
# so three and four bits are where the two models can be put side by side.
# Same commit rule on both sides: confidence 0.95.
#
#   tmux new -s fd5 -d "bash scripts/pod/fd5.sh 2>&1 | tee out/_fd5.log"

. "$(dirname "$0")/lib.sh"

MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B"
NEEDFD=34000
EVALFD="bash scripts/llada2.sh scripts/evaluate_fdv2.py $MFD"
C="--n-eval 200 --gen-length 2048 --threshold 0.95 --block-size 32 --small-block-size 8"
e() { n=$1; shift; run "$n" $NEEDFD "out/$n.json" $EVALFD $C "$@" --out "out/$n.json"; }

# three bits: the width where LLaDA2.0-mini's axis effect first shows
e fd_3_ch     --kv-bits 3
e fd_3_tok    --kv-bits 3 --kv-key-axis channel
e fd_3_quarot --kv-bits 3 --kv-key-axis channel --rotate-qk
# four bits: the axis on both sides of LLaDA2.0-mini's free width
e fd_4_tok    --kv-bits 4 --kv-key-axis channel
e fd_4_quarot --kv-bits 4 --kv-key-axis channel --rotate-qk

echo "$(date +%H:%M) Fast-dLLM-v2 против MoE, оси при 3 и 4 битах: готово"
