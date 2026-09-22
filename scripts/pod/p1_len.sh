# Part 1, thesis 4: is Fast-dLLM-v2's sensitivity a matter of generation length?
#
# At four bits per token the centered logit error is the same on LLaDA2.0-mini (0.351) and
# Fast-dLLM-v2 (0.346), and one keeps 91.5 while the other falls to 0.0. The candidate is
# length: LLaDA2.0-mini generates 512 tokens, Fast-dLLM-v2 up to 2048 of its own reasoning,
# and every cache error is read by all the blocks after it. Cut Fast-dLLM-v2 to 512 and look
# at the cells that separate the models. If per token at four bits survives at 512, length is
# the reason; if it still dies, the models differ in something else.
#
#   tmux new -s p1b -d "bash scripts/pod/p1_len.sh 2>&1 | tee out/_p1b.log"

. "$(dirname "$0")/lib.sh"

MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B"
NEEDFD=34000
EVALFD="bash scripts/llada2.sh scripts/evaluate_fdv2.py $MFD"
C="--n-eval 200 --gen-length 512 --threshold 0.95 --block-size 32 --small-block-size 8"
e() { n=$1; shift; run "$n" $NEEDFD "out/$n.json" $EVALFD $C "$@" --out "out/$n.json"; }

e fd512_16      --kv-bits 16
e fd512_4_tok   --kv-bits 4 --kv-key-axis channel
e fd512_4_ch    --kv-bits 4
e fd512_3_ch    --kv-bits 3

echo "$(date +%H:%M) часть 1: Fast-dLLM-v2 при 512 токенах — готово"
