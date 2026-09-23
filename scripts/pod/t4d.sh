# Does the movement measure predict the intervention it never saw?
#
# The KL-to-accuracy curve inside Fast-dLLM-v2 was built from cells with nothing done to
# them. Taking the bias out of the store moves three of them a long way -- 74.0 to 85.0 per
# channel at four bits, 18.0 to 43.5 at two, 1.5 to 79.0 under rotation -- and if KL is the
# quantity the accuracy follows, those cells have to land on the same curve at their new
# movement. If they land far off it, KL orders widths within one tensor and nothing more.
#
# Two dumps, both with the metrics the plain one now carries: the exact removal and the
# mean, so the two-bit gap between them (43.5 against 28.0, p = 2e-5) can be read as
# movement as well.
#
#   tmux new -s t4d -d "bash scripts/pod/t4d.sh 2>&1 | tee out/_t4d.log"
#   python scripts/diag_compare.py out/ke_afd.json out/ke_apb.json out/ke_amean.json --centered --bits 2

. "$(dirname "$0")/lib.sh"

D="--samples 16 --bits 4 3 2 --mask-ratio 0.5 --all-layers --rotate --logit-error"
MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B --model-type fast_dllm_v2"

run ke_apb   24000 out/ke_apb.json   $PY20 scripts/check_key_error.py $MFD $D \
    --pre-bias --dump out/ke_apb.json
run ke_amean 24000 out/ke_amean.json $PY20 scripts/check_key_error.py $MFD $D \
    --key-mean --dump out/ke_amean.json

echo "$(date +%H:%M) движение внимания под вмешательством — готово"
