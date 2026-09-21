# Fast-dLLM-v2: does keeping the k_proj bias out of the store rescue the axis at four bits?
#
# At four bits per channel holds (74.0) and per token and QuaRot die (0.0, ~1). If the bias
# is what makes the wrong axis fatal here, --pre-bias -- exact at any width -- should bring
# them back, the way the gain migration brought back LLaDA2.0-mini's per-token cache.
#
#   tmux new -s fd6 -d "bash scripts/pod/fd6.sh 2>&1 | tee out/_fd6.log"

. "$(dirname "$0")/lib.sh"

MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B"
NEEDFD=34000
EVALFD="bash scripts/llada2.sh scripts/evaluate_fdv2.py $MFD"
C="--n-eval 200 --gen-length 2048 --threshold 0.95 --block-size 32 --small-block-size 8"
e() { n=$1; shift; run "$n" $NEEDFD "out/$n.json" $EVALFD $C "$@" --out "out/$n.json"; }

e fd_16_pb        --kv-bits 16 --pre-bias
e fd_4_tok_pb     --kv-bits 4 --kv-key-axis channel --pre-bias
e fd_4_quarot_pb  --kv-bits 4 --kv-key-axis channel --rotate-qk --pre-bias
e fd_4_ch_pb      --kv-bits 4 --pre-bias

echo "$(date +%H:%M) Fast-dLLM-v2, bias вне кэша при 4 битах: готово"
