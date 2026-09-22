# Fast-dLLM-v2: exact removal of the rotated bias against SageAttention's smooth K.
#
# Taking a key-shared component out before the quantizer is prior art: SageAttention
# subtracts the per-channel mean of the keys before a per-token scale. Our form subtracts
# RoPE(b) at its own position, exact and without data. If the mean does as well, the exact
# form adds nothing and the right thing is to use SageAttention's with a citation. Each cell
# here has a --pre-bias twin already measured:
#   fd_4_tok_pb 84.0   fd_3_ch_pb 80.5   fd_3_tok_pb 76.5   fd_4_quarot_pb2 79.0
# and the plain cells 0.0 / 58.0 / 0.0 / 1.5.
# The tensor run first: minutes, and it shows per layer where the two differ.
#
#   tmux new -s sage -d "bash scripts/pod/fd_sage.sh 2>&1 | tee out/_sage.log"
#   python scripts/diag_compare.py out/ke_cfd.json out/ke_cfd_pb2.json out/ke_cfd_mean.json --centered --bits 4

. "$(dirname "$0")/lib.sh"

MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B"
D="--samples 16 --bits 4 3 2 --mask-ratio 0.5 --all-layers --rotate --logit-error"
run ke_cfd_mean 24000 out/ke_cfd_mean.json $PY20 scripts/check_key_error.py $MFD \
    --model-type fast_dllm_v2 $D --key-mean --dump out/ke_cfd_mean.json

NEEDFD=34000
EVALFD="bash scripts/llada2.sh scripts/evaluate_fdv2.py $MFD"
C="--n-eval 200 --gen-length 2048 --threshold 0.95 --block-size 32 --small-block-size 8"
e() { n=$1; shift; run "$n" $NEEDFD "out/$n.json" $EVALFD $C "$@" --out "out/$n.json"; }

e fd_4_tok_mean    --kv-bits 4 --kv-key-axis channel --key-mean
e fd_3_ch_mean     --kv-bits 3 --key-mean
e fd_3_tok_mean    --kv-bits 3 --kv-key-axis channel --key-mean
e fd_4_quarot_mean --kv-bits 4 --kv-key-axis channel --rotate-qk --key-mean

echo "$(date +%H:%M) Fast-dLLM-v2: точное удаление bias против среднего (SageAttention) — готово"
