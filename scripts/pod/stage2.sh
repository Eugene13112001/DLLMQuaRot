# Stage 2: move the K-norm gain into the Q-norm and see whether the two-bit collapse goes with it.
# The transform leaves attention exactly as it was (the factor is constant on every rotary pair),
# so the 16-bit row is the control: it must reproduce the untouched model.
#
#   tmux new -s s2 -d "bash scripts/pod/stage2.sh 2>&1 | tee out/_s2.log"

. "$(dirname "$0")/lib.sh"

C="--n-eval 200 --gen-length 512 --eval-steps 256 --kv-cache --kv-policy block"
e() { n=$1; shift; run "$n" $NEED20 "out/$n.json" $EVAL20 $C "$@" --out "out/$n.json"; }

# control: the migration must not move the model on its own
e l20_M_16      --kv-bits 16 --migrate-qk 1.0
# the collapse cells, repeated with the gain moved out of the keys
e l20_M2_tok    --kv-bits 2 --kv-key-axis channel --migrate-qk 1.0
e l20_M2_quarot --kv-bits 2 --kv-key-axis channel --rotate-qk --migrate-qk 1.0
e l20_M2_ch     --kv-bits 2 --migrate-qk 1.0
# half migration, to see whether the effect moves with alpha
e l20_M2_tok_a5 --kv-bits 2 --kv-key-axis channel --migrate-qk 0.5

# the same on the tensor: the axis ratio should fall towards LLaDA-1.5's
run ke_l20_mig $NEED20 out/ke_l20_mig.json $PY20 scripts/check_key_error.py $M20 \
    --samples 32 --bits 4 3 2 --mask-ratio 0.5 --all-layers --split-rope --rotate \
    --migrate-qk 1.0 --dump out/ke_l20_mig.json

echo "$(date +%H:%M) этап 2 прошёл"
