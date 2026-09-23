# Part 1, thesis 4, step 3: equal movement, on answers.
#
# The tensor run split the gap in two. At four bits per token the two models carry the
# same centered logit error (0.351 and 0.346) but not the same movement: Fast-dLLM-v2's
# attention travels twice as far (KL 0.470 against 0.222), because its attention is the
# most peaked of the three (24.2 effective keys against 32.2 and 55.9). That is the first
# factor, and it is measured. The second is what is left after it: at KL 0.8
# LLaDA2.0-mini still answers 59.5 while Fast-dLLM-v2 is at 0.5 with half that movement.
#
# So this is a 2x2: both models at both movement levels, with structureless noise that
# carries the movement and none of the quantizer's shape.
#
#   python scripts/noise_dose.py out/kn_*.json --target-kl 0.47 0.22
#   tmux new -s t4c -d "S20_47=.. SFD_47=.. S20_22=.. SFD_22=.. bash scripts/pod/t4c.sh 2>&1 | tee out/_t4c.log"
#
# Against: the quantizer's own cells at those movements -- LLaDA2.0-mini 91.5 at KL 0.222
# and Fast-dLLM-v2 0.0 at KL 0.470 -- and the references at 16 bits, 93.5 and 82.0.
#
#   noise reproduces each model's quantized cell -> the shape of the error carries nothing
#     the movement does not, and thesis 4 is two measured factors: geometry, then tolerance.
#   Fast-dLLM-v2 survives the noise it dies of when quantized -> the shape is what kills,
#     and the loud channels are where to look for it.
#   LLaDA2.0-mini dies of noise at KL 0.222 -> the movement is not what its quantizer's
#     error does to it, and the tensor measure does not transfer to answers at all.

. "$(dirname "$0")/lib.sh"

: "${S20_47:?set the four doses from scripts/noise_dose.py --target-kl 0.47 0.22}"
: "${SFD_47:?}"; : "${S20_22:?}"; : "${SFD_22:?}"

C20="--n-eval 200 --gen-length 512 --eval-steps 256 --kv-cache --threshold 0.95 --kv-bits 16"
e20() { n=$1; shift; run "$n" $NEED20 "out/$n.json" $EVAL20 $C20 "$@" --out "out/$n.json"; }

MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B"
EVALFD="bash scripts/llada2.sh scripts/evaluate_fdv2.py $MFD"
CFD="--n-eval 200 --gen-length 512 --threshold 0.95 --block-size 32 --small-block-size 8 --kv-bits 16"
efd() { n=$1; shift; run "$n" 34000 "out/$n.json" $EVALFD $CFD "$@" --out "out/$n.json"; }

# The level Fast-dLLM-v2's quantizer dies at, given to both models.
efd fd_noise_kl47 --kv-key-noise "$SFD_47"
e20 l20_noise_kl47 --kv-key-noise "$S20_47"
# The level LLaDA2.0-mini's quantizer carries unharmed, given to both.
e20 l20_noise_kl22 --kv-key-noise "$S20_22"
efd fd_noise_kl22 --kv-key-noise "$SFD_22"

echo "$(date +%H:%M) тезис 4: равное движение внимания на ответах — готово"
