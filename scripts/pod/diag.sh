# Why QuaRot is not rescued, and what the gain migration leaves behind -- on the tensor.
#
# Two bits (and three for scale), 16 canvases, every layer, logit error with the split into
# the channels the K-norm gain makes loud and the rest, plus the q-k geometry that the
# isotropic-noise explanation of QuaRot predicts from. A dose sweep of the migration on
# LLaDA2.0-mini and LLaDA-1.5 for comparison. Minutes per run on a free card.
#
#   tmux new -s diag -d "bash scripts/pod/diag.sh 2>&1 | tee out/_diag.log"
#   python scripts/diag_compare.py out/ke_d20_a0.json out/ke_d20_a025.json out/ke_d20_a05.json \
#       out/ke_d20_a075.json out/ke_d20_a1.json out/ke_d20_geo.json out/ke_d15.json

. "$(dirname "$0")/lib.sh"

D="--samples 16 --bits 2 3 --mask-ratio 0.5 --all-layers --rotate --logit-error"
k20() { n=$1; shift; run "$n" $NEED20 "out/$n.json" $PY20 scripts/check_key_error.py $M20 $D "$@" --dump "out/$n.json"; }

k20 ke_d20_a0
k20 ke_d20_a1   --migrate-qk 1.0
k20 ke_d20_a05  --migrate-qk 0.5
k20 ke_d20_a075 --migrate-qk 0.75
k20 ke_d20_a025 --migrate-qk 0.25
k20 ke_d20_geo  --migrate-qk 1.0 --migrate-mode geomean
run ke_d15 $NEED15 out/ke_d15.json $PY15 scripts/check_key_error.py $M15 $D --dump out/ke_d15.json

echo "$(date +%H:%M) диагностика: готово"
