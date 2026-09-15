# Stage 1, the weight law: every layer, rotated and unrotated half of the head split, QuaRot cells, 4/3/2 bits.
# Minutes per dump once a card is free. Same idempotent runner as queue.sh.
#
#   tmux new -s law -d "bash scripts/pod/law.sh 2>&1 | tee out/_law.log"
#   python scripts/gamma_law.py out/law_l20.json --nonorm out/law_l20_nonorm.json --bits 4

. "$(dirname "$0")/lib.sh"

A="--samples 32 --bits 4 3 2 --mask-ratio 0.5 --all-layers --split-rope --rotate"
run law_l20        $NEED20 out/law_l20.json        $PY20 scripts/check_key_error.py $M20 $A --dump out/law_l20.json
run law_l20_nonorm $NEED20 out/law_l20_nonorm.json $PY20 scripts/check_key_error.py $M20 $A --skip-qk-norm --dump out/law_l20_nonorm.json
run law_l15        $NEED15 out/law_l15.json        $PY15 scripts/check_key_error.py $M15 $A --dump out/law_l15.json

echo "$(date +%H:%M) закон по весам: замеры готовы"
