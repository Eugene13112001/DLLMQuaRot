# Fast-dLLM-v2 at two bits: with the bias out of the store, is the remaining loss K or V?
#
# Per channel at 2 bits the bias intervention lifts 18.0 to 43.5, still far from 82.5.
# Keys alone and values alone at 2 bits split that residue. If it is V, the two-bit story
# on this model is about values, not about the key outlier, and has to be told separately.
# The third cell only matters if V is the culprit: the same values at BitSieve's 32-channel
# groups instead of the whole 128-channel head.
#
#   tmux new -s fd10 -d "bash scripts/pod/fd10.sh 2>&1 | tee out/_fd10.log"

. "$(dirname "$0")/lib.sh"

MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B"
NEEDFD=34000
EVALFD="bash scripts/llada2.sh scripts/evaluate_fdv2.py $MFD"
C="--n-eval 200 --gen-length 2048 --threshold 0.95 --block-size 32 --small-block-size 8"
e() { n=$1; shift; run "$n" $NEEDFD "out/$n.json" $EVALFD $C "$@" --out "out/$n.json"; }

e fd_2k_pb   --kv-bits 16 --kv-key-bits 2 --pre-bias
e fd_2v      --kv-bits 16 --kv-value-bits 2
e fd_2v_g32  --kv-bits 16 --kv-value-bits 2 --kv-value-group-size 32

echo "$(date +%H:%M) Fast-dLLM-v2, остаток при 2 битах (K или V): готово"
