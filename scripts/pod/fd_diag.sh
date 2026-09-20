# Fast-dLLM-v2 on the tensor: the same measurements the LLaDA pair has.
#
# Reads the keys the cache stores and reports, per layer: the cost of the wrong axis, the
# cost of QuaRot, the crest factor, the relative error of the attention logits, and the
# q-k geometry (|cos| and the isotropic-noise factor) that explains QuaRot elsewhere.
# Directly comparable to out/ke_d20_*.json (LLaDA2.0-mini) and out/ke_d15.json (LLaDA-1.5),
# because it is the same script with the same grid.
#
# The prediction worth writing down first: this model has no QK-Norm, so if its keys still
# carry fixed-channel outliers they come from the bias on k_proj, and the axis ratio should
# sit between LLaDA-1.5 (1.9x, no parameter-induced outlier) and LLaDA2.0-mini (4.8x).
#
#   tmux new -s fdd -d "bash scripts/pod/fd_diag.sh 2>&1 | tee out/_fdd.log"
#   python scripts/diag_compare.py out/ke_dfd.json out/ke_d20_a0.json out/ke_d15.json

. "$(dirname "$0")/lib.sh"

MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B --model-type fast_dllm_v2"
NEEDFD=24000
D="--samples 16 --bits 4 3 2 --mask-ratio 0.5 --all-layers --rotate --logit-error"

run ke_dfd $NEEDFD out/ke_dfd.json $PY20 scripts/check_key_error.py $MFD $D --dump out/ke_dfd.json

echo "$(date +%H:%M) Fast-dLLM-v2 на тензоре: готово"
