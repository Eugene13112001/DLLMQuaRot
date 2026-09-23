# The two objections our explanations face, measured on all three models.
#
# 1. KVTuner reports heads with *sparse* attention as the more robust to cache
#    quantization -- the opposite sign to ours, where concentrated attention is the
#    more fragile. The reconciliation on offer is the attention sink: if the first few
#    positions hold the mass, perturbing that distribution reorders nothing. If our low
#    effective-key count (24.2 against 32.2 and 55.9) is the same sink, the explanation
#    is theirs and not ours. This measures the mass on the first four allowed keys and
#    the effective count with those dropped and the rest re-softmaxed.
#
# 2. Zero-mean noise on the stored keys multiplies their softmax weight by exp(Var/2),
#    because exp is convex. A factor every key carries cancels -- ours does not, since
#    the prefix is quantized while the current block is recomputed clean, so the prefix
#    is inflated against the block. That is a rival explanation for noise being worse
#    than the quantizer on Fast-dLLM-v2 (3.0 against 43.5-58.0 at a matched movement).
#    The run prints the term per unit variance and, at the four doses we used, its size
#    as a share of the logit spread.
#
#   tmux new -s obj -d "bash scripts/pod/obj.sh 2>&1 | tee out/_obj.log"
#   grep -A 14 "две возражения\|the two objections" out/ke_o*.log
#
# Reading it: effective count without the sink close to the count with it -> the
# concentration is not the sink, KVTuner's sign is about something else, our claim
# stands. Jensen term a few percent of the spread or more -> it is a live rival and has
# to be subtracted before the noise result can be attributed to the model.

. "$(dirname "$0")/lib.sh"

D="--samples 16 --bits 4 --mask-ratio 0.5 --all-layers --logit-error --sink-keys 4"
MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B --model-type fast_dllm_v2"

run ke_ofd 24000   out/ke_ofd.json $PY20 scripts/check_key_error.py $MFD $D --dump out/ke_ofd.json
run ke_o20 $NEED20 out/ke_o20.json $PY20 scripts/check_key_error.py $M20 $D --dump out/ke_o20.json
run ke_o15 $NEED15 out/ke_o15.json $PY15 scripts/check_key_error.py $M15 $D --dump out/ke_o15.json

echo "$(date +%H:%M) сток внимания и смещение Йенсена: готово"
