# Does the exact-minus-mean gap follow the number of positions sharing one mean?
#
# The thesis says a mean over positions keeps only the part of the bias that does not
# rotate. If that is right, the gap between exact RoPE(b) subtraction and the mean is not
# a bare budget crossover -- it must grow with the number of positions the mean is taken
# over, because a longer stretch of positions spans more rotation.
#
# The mean is taken per cache write (`m = k.mean(dim=-2)` in the quantized cache), so the
# knob is the write length, i.e. the block size -- NOT --kv-group-size, which collapses to
# the write length whenever the group is longer (g = min(group_size, dim)); at blocks of 32
# tokens a group of 128 and a group of 32 are already the same quantizer.
#
# At the shipped block size of 32 all three arms exist: plain 18.0, mean 28.0, exact 43.5,
# a gap of 15.5 points (45:14, p = 2e-5). Two more block sizes turn that single number into
# a slope. Prediction, stated before the runs:
#
#   block 8  -> gap shrinks toward zero   (few positions per mean, little rotation spanned)
#   block 64 -> gap wider than 15.5
#
# A flat or inverted slope kills the positional reading and makes us another instance of
# the budget crossover in 2605.08114. Block size shifts accuracy by itself, which is why
# the quantity read is the *gap* -- both arms carry the same block size -- and why the
# no-removal controls are here too, to show the quantizer is stressed at all.
#
# Part C is the tensor probe for the partial-RoPE section, minutes on a free card, and it
# needs no generation: the gamma spread and the axis cost split into the rotated and the
# unrotated half of the head.
#
#   tmux new -s pe -d "bash scripts/pod/pe.sh 2>&1 | tee out/_pe.log"

. "$(dirname "$0")/lib.sh"

# ---------------------------------------------------------------- C: tensor, minutes
D="--samples 32 --bits 4 2 --mask-ratio 0.5 --all-layers --split-rope --rotate --logit-error"
run ke_pe20 $NEED20 out/ke_pe20.json \
    $PY20 scripts/check_key_error.py $M20 $D --dump out/ke_pe20.json

# ---------------------------------------------------------------- A: the slope
MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B"
EVALFD="bash scripts/llada2.sh scripts/evaluate_fdv2.py $MFD"
C="--n-eval 200 --gen-length 2048 --threshold 0.95 --small-block-size 8"
e() { n=$1; shift; run "$n" 34000 "out/$n.json" $EVALFD $C "$@" --out "out/$n.json"; }

# Short writes first: if the gap is already gone at 8, the slope is decided with two runs.
e fd_2_b8_pre   --kv-bits 2 --block-size 8  --pre-bias
e fd_2_b8_mean  --kv-bits 2 --block-size 8  --key-mean

# Long writes. Fast-dLLM-v2 ships block 32 with sub-blocks of 8; 64 is outside the shipped
# setting and the model code may refuse it, in which case this dies fast and costs little.
e fd_2_b64_pre  --kv-bits 2 --block-size 64 --pre-bias
e fd_2_b64_mean --kv-bits 2 --block-size 64 --key-mean

# Controls: no removal at either block size, so a small gap can be told apart from a
# quantizer that was never stressed.
e fd_2_b8      --kv-bits 2 --block-size 8
e fd_2_b64     --kv-bits 2 --block-size 64

# Optional: the window PatternKV refreshes at (every 128 tokens), so the comparison against
# a published piecewise pattern is direct rather than by argument. Furthest from the shipped
# setting, so it is last -- if the model code refuses 64 it will refuse this too.
e fd_2_b128_pre  --kv-bits 2 --block-size 128 --pre-bias
e fd_2_b128_mean --kv-bits 2 --block-size 128 --key-mean

echo "$(date +%H:%M) наклон разрыва по длине записи — готово"
