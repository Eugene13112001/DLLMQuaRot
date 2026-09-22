# Part 1, thesis 4, step 3: the matched dose on answers.
#
# Set the two doses from the calibration sweep first:
#   python scripts/noise_dose.py out/kn_*.json --target 0.35
#   tmux new -s t4c -d "S20=0.10 SFD=0.11 bash scripts/pod/t4c.sh 2>&1 | tee out/_t4c.log"
#
# Four cells against numbers already measured. The quantizer's four-bit per-token cell:
# LLaDA2.0-mini 91.5, Fast-dLLM-v2 0.0 (at 512 tokens too). The references at 16 bits:
# 93.5 and 82.0. The noise carries the same centered logit error and none of the shape.
#
#   both survive the noise -> the shape of the quantizer's error is what kills, and the
#     tensor run (t4) says which shape: the movement per unit error, KL/err^2.
#   both die -> the size is what kills, and LLaDA2.0-mini's survival at four bits is a
#     property of its quantizer's error, not of the model.
#   the gap survives the noise -> the models differ in sensitivity to any error of that
#     size; the gap is the model's, and thesis 4 becomes a measured constant per model.
#
# Half doses are included because a control that lands on the wrong side of the cliff
# tells us the cliff exists but not where -- two points per model give a slope.

. "$(dirname "$0")/lib.sh"

: "${S20:?set S20, the LLaDA2.0-mini dose from scripts/noise_dose.py}"
: "${SFD:?set SFD, the Fast-dLLM-v2 dose from scripts/noise_dose.py}"
H20=$(awk -v s="$S20" 'BEGIN{printf "%.4f", s / 2}')
HFD=$(awk -v s="$SFD" 'BEGIN{printf "%.4f", s / 2}')

C20="--n-eval 200 --gen-length 512 --eval-steps 256 --kv-cache --threshold 0.95 --kv-bits 16"
e20() { n=$1; shift; run "$n" $NEED20 "out/$n.json" $EVAL20 $C20 "$@" --out "out/$n.json"; }

MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B"
EVALFD="bash scripts/llada2.sh scripts/evaluate_fdv2.py $MFD"
CFD="--n-eval 200 --gen-length 512 --threshold 0.95 --block-size 32 --small-block-size 8 --kv-bits 16"
efd() { n=$1; shift; run "$n" 34000 "out/$n.json" $EVALFD $CFD "$@" --out "out/$n.json"; }

efd fd_noise_full --kv-key-noise "$SFD"
e20 l20_noise_full --kv-key-noise "$S20"
efd fd_noise_half --kv-key-noise "$HFD"
e20 l20_noise_half --kv-key-noise "$H20"

echo "$(date +%H:%M) тезис 4: бесструктурная ошибка той же величины — готово"
