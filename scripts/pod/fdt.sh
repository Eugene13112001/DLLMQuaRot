# Fast-dLLM-v2 on the tensor with the bias kept out of the store -- the gate for the bias
# intervention, as check_migration.py is for the gain: does the crest of the stored keys
# fall, and does the logit error of each axis fall with it? Also records k_bias in the dump
# for scripts/bias_law.py.
#
#   tmux new -s fdt -d "bash scripts/pod/fdt.sh 2>&1 | tee out/_fdt.log"
#   python scripts/diag_compare.py out/ke_cfd.json out/ke_cfd_pb.json --centered --bits 4
#   python scripts/bias_law.py out/ke_cfd_pb.json --bits 4

. "$(dirname "$0")/lib.sh"

MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B --model-type fast_dllm_v2"
D="--samples 16 --bits 4 3 2 --mask-ratio 0.5 --all-layers --rotate --logit-error"

run ke_cfd_pb 24000 out/ke_cfd_pb.json $PY20 scripts/check_key_error.py $MFD $D --pre-bias --dump out/ke_cfd_pb.json

echo "$(date +%H:%M) Fast-dLLM-v2, тензор без bias: готово"
