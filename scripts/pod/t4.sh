# Part 1, thesis 4: the same error, a different answer -- what does attention do with it?
#
# At four bits per token LLaDA2.0-mini and Fast-dLLM-v2 carry the same centered logit error
# (0.351 and 0.346) and score 91.5 and 0.0. Generation length is not the reason (p1_len).
# Neither is the spread of the logits: the centered error already *is* the error in units of
# that spread, exactly (tests/test_attn_divergence.py pins the identity). What is left on
# the tensor is the movement itself -- how far the attention distribution travels for an
# error of that size, and how peaked the distribution it travels from is. This re-takes the
# three probes with KL, the share of queries whose top-1 key changes, and exp(entropy) of
# the true attention.
#
#   tmux new -s t4 -d "bash scripts/pod/t4.sh 2>&1 | tee out/_t4.log"
#   python scripts/diag_compare.py out/ke_a20.json out/ke_afd.json out/ke_a15.json --centered --bits 4

. "$(dirname "$0")/lib.sh"

D="--samples 16 --bits 4 3 2 --mask-ratio 0.5 --all-layers --rotate --logit-error"
MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B --model-type fast_dllm_v2"

run ke_afd 24000   out/ke_afd.json $PY20 scripts/check_key_error.py $MFD $D --dump out/ke_afd.json
run ke_a20 $NEED20 out/ke_a20.json $PY20 scripts/check_key_error.py $M20 $D --dump out/ke_a20.json
run ke_a15 $NEED15 out/ke_a15.json $PY15 scripts/check_key_error.py $M15 $D --dump out/ke_a15.json

echo "$(date +%H:%M) тезис 4: движение внимания при равной ошибке — готово"
