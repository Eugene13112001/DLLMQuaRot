# Fast-dLLM-v2: does keeping the k_proj bias out of the store rescue 3 and 2 bits too?
#
# At 4 bits it does, completely: per token 0.0 -> 84.0, per channel 74.0 -> 85.0. Below 4
# bits the model is weak even on the right axis (58.0 at 3, 18.0 at 2); if the bias is what
# makes it so, the same exact intervention should lift those cells as well, the way the gain
# migration was tested at 2 bits on LLaDA2.0-mini. Three bits first -- they answer the
# question at a width the model may survive; the 2-bit cells are slow if it does not.
#
#   tmux new -s fd7 -d "bash scripts/pod/fd7.sh 2>&1 | tee out/_fd7.log"

. "$(dirname "$0")/lib.sh"

MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B"
NEEDFD=34000
EVALFD="bash scripts/llada2.sh scripts/evaluate_fdv2.py $MFD"
C="--n-eval 200 --gen-length 2048 --threshold 0.95 --block-size 32 --small-block-size 8"
e() { n=$1; shift; run "$n" $NEEDFD "out/$n.json" $EVALFD $C "$@" --out "out/$n.json"; }

e fd_3_ch_pb     --kv-bits 3 --pre-bias
e fd_3_tok_pb    --kv-bits 3 --kv-key-axis channel --pre-bias
e fd_3_quarot_pb --kv-bits 3 --kv-key-axis channel --rotate-qk --pre-bias
e fd_2_ch_pb     --kv-bits 2 --pre-bias
e fd_2_tok_pb    --kv-bits 2 --kv-key-axis channel --pre-bias

echo "$(date +%H:%M) Fast-dLLM-v2, bias вне кэша при 3 и 2 битах: готово"
