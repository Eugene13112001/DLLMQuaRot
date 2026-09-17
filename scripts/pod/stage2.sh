# Stage 2: move the K-norm gain into the Q-norm and see whether the two-bit collapse goes with it.
# The transform leaves attention exactly as it was (the factor is constant on every rotary pair),
# so the 16-bit row is the control: it must reproduce the untouched model.
#
#   tmux new -s s2 -d "bash scripts/pod/stage2.sh 2>&1 | tee out/_s2.log"

. "$(dirname "$0")/lib.sh"

# Gate: one forward pass says whether the migration is the identity it claims to be. If the
# attention probabilities move by more than rounding, nothing below is worth a card.
run mig_shrink $NEED20 out/mig_shrink.done $PY20 scripts/check_migration.py $M20 --alpha 1.0 --done out/mig_shrink.done
if [ ! -s out/mig_shrink.done ]; then
  echo "$(date +%H:%M) !!! перенос множителей не прошёл проверку, см. out/mig_shrink.log"
  exit 1
fi

C="--n-eval 200 --gen-length 512 --eval-steps 256 --kv-cache --kv-policy block"
e() { n=$1; shift; run "$n" $NEED20 "out/$n.json" $EVAL20 $C "$@" --out "out/$n.json"; }

# the collapse cells, repeated with the gain moved out of the keys
e l20_M2_tok    --kv-bits 2 --kv-key-axis channel --migrate-qk 1.0
e l20_M2_quarot --kv-bits 2 --kv-key-axis channel --rotate-qk --migrate-qk 1.0
e l20_M2_ch     --kv-bits 2 --migrate-qk 1.0
# control: the migration must not move the model on its own. The gate already measured the
# attention drift as exactly zero on every layer, so this is the end-to-end confirmation and goes
# after the cells that answer the question.
e l20_M_16      --kv-bits 16 --migrate-qk 1.0
# half migration, to see whether the effect moves with alpha
e l20_M2_tok_a5 --kv-bits 2 --kv-key-axis channel --migrate-qk 0.5

# the same on the tensor, read as attention reads it. The error in K space is not
# comparable across the migration (the loud channels move into Q and multiply
# whatever error K carries there); the logit error q (K - Q(K))^T is, because the
# migration leaves q K^T bit-identical. Both runs carry it, so the pair is one
# variable apart.
T="--samples 32 --bits 4 3 2 --mask-ratio 0.5 --all-layers --rotate --logit-error"
run ke_l20_base_L $NEED20 out/ke_l20_base_L.json $PY20 scripts/check_key_error.py $M20 $T     --dump out/ke_l20_base_L.json
run ke_l20_mig_L  $NEED20 out/ke_l20_mig_L.json  $PY20 scripts/check_key_error.py $M20 $T     --migrate-qk 1.0 --dump out/ke_l20_mig_L.json

echo "$(date +%H:%M) этап 2 прошёл"
