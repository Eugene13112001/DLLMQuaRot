# Fast-dLLM-v2, second pass: the intervention and the widths in between.
#
# fd.sh answers whether the key axis decides this cache at two bits. If it does, the cause
# cannot be a QK-Norm gain -- there is none -- and the candidate is the bias on k_proj, the
# mechanism reported for Qwen2.5 elsewhere. --pre-bias is the test of that: the stored key
# keeps the rotated bias out of the quantizer's range and gets it back on read, which is
# exact at any width. It is to this model what the gain migration is to LLaDA2.0-mini.
#
#   tmux new -s fd2 -d "bash scripts/pod/fd2.sh 2>&1 | tee out/_fd2.log"

. "$(dirname "$0")/lib.sh"

MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B"
NEEDFD=34000
EVALFD="bash scripts/llada2.sh scripts/evaluate_fdv2.py $MFD"
C="--n-eval 200 --gen-length 2048 --threshold 0.95 --block-size 32 --small-block-size 8"
e() { n=$1; shift; run "$n" $NEEDFD "out/$n.json" $EVALFD $C "$@" --out "out/$n.json"; }

# control: the intervention must not move a lossless run at all
e fd_16_pb      --kv-bits 16 --pre-bias
# the collapsing cells, repeated with the bias kept out of the store
e fd_2_tok_pb   --kv-bits 2 --kv-key-axis channel --pre-bias
e fd_2_quarot_pb --kv-bits 2 --kv-key-axis channel --rotate-qk --pre-bias
e fd_2_ch_pb    --kv-bits 2 --pre-bias
# three bits, to see where this model's floor is
e fd_3_ch       --kv-bits 3
e fd_3_tok      --kv-bits 3 --kv-key-axis channel

echo "$(date +%H:%M) Fast-dLLM-v2, вмешательство: готово"
