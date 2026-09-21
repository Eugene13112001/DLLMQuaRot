# The logit error again, centered per query -- the part softmax actually sees.
#
# The uncentered ratio called Fast-dLLM-v2 the robust model (axis cost 2.3x against 4.9x,
# QuaRot 0.21 against 0.45), and on answers it is the fragile one: per token and QuaRot die
# at four bits there, where every axis lives on LLaDA2.0-mini. A k_proj bias adds nearly the
# same vector to every key, so q . b is close to one constant per query; it inflates
# ||q K^T|| without moving a single attention weight. The centered error removes each
# query's mean over the keys it may read before comparing. Every model is re-taken here
# with both numbers in the dump, so the old and the new metric sit side by side.
#
#   tmux new -s cdiag -d "bash scripts/pod/cdiag.sh 2>&1 | tee out/_cdiag.log"
#   python scripts/diag_compare.py out/ke_c20.json out/ke_c20m.json out/ke_cfd.json out/ke_c15.json --centered --bits 4

. "$(dirname "$0")/lib.sh"

D="--samples 16 --bits 4 3 2 --mask-ratio 0.5 --all-layers --rotate --logit-error"
MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B --model-type fast_dllm_v2"

run ke_cfd  24000   out/ke_cfd.json  $PY20 scripts/check_key_error.py $MFD $D --dump out/ke_cfd.json
run ke_c20  $NEED20 out/ke_c20.json  $PY20 scripts/check_key_error.py $M20 $D --dump out/ke_c20.json
run ke_c20m $NEED20 out/ke_c20m.json $PY20 scripts/check_key_error.py $M20 $D --migrate-qk 1.0 --dump out/ke_c20m.json
run ke_c15  $NEED15 out/ke_c15.json  $PY15 scripts/check_key_error.py $M15 $D --dump out/ke_c15.json

echo "$(date +%H:%M) центрированная ошибка логитов: готово"
