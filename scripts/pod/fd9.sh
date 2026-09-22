# Fast-dLLM-v2: QuaRot with the bias kept out of the store -- the cells fd6/fd7 got wrong.
#
# Under R4 the keys reaching the store are RoPE(Wx + b) H. Until 8b96142 the pre-bias cache
# subtracted RoPE(b) without H: still an identity, so the lossless control could not catch it,
# but the bias stayed inside the quantizer's range. fd_4_quarot_pb (0.0) and fd_3_quarot_pb
# therefore measured plain QuaRot, not QuaRot with the bias out. On the tensor, where the
# order was right, the centered QuaRot error falls from 0.271 to 0.093 -- below the per-token
# cell that answers 84.0 -- so whether rotation is rescued is an open question again.
# Also re-takes the tensor with ||RoPE(b)|| / ||k|| recorded for bias_law.py.
#
#   tmux new -s fd9 -d "bash scripts/pod/fd9.sh 2>&1 | tee out/_fd9.log"

. "$(dirname "$0")/lib.sh"

MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B"
NEEDFD=34000
EVALFD="bash scripts/llada2.sh scripts/evaluate_fdv2.py $MFD"
C="--n-eval 200 --gen-length 2048 --threshold 0.95 --block-size 32 --small-block-size 8"
e() { n=$1; shift; run "$n" $NEEDFD "out/$n.json" $EVALFD $C "$@" --out "out/$n.json"; }

D="--samples 16 --bits 4 3 2 --mask-ratio 0.5 --all-layers --rotate --logit-error"
run ke_cfd_pb2 24000 out/ke_cfd_pb2.json $PY20 scripts/check_key_error.py $MFD \
    --model-type fast_dllm_v2 $D --pre-bias --dump out/ke_cfd_pb2.json

e fd_4_quarot_pb2 --kv-bits 4 --kv-key-axis channel --rotate-qk --pre-bias
e fd_3_quarot_pb2 --kv-bits 3 --kv-key-axis channel --rotate-qk --pre-bias

echo "$(date +%H:%M) Fast-dLLM-v2, QuaRot без bias (исправлено): готово"
