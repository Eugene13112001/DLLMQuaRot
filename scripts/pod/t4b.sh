# Part 1, thesis 4, step 2: calibrate the structureless control.
#
# The two models agree on the centered logit error at four bits per token (0.351 and
# 0.346) and score 91.5 and 0.0. Either the *shape* of the quantizer's error is what
# separates them, or one of them dies of any error of that size. Gaussian noise on the
# stored keys has no shape, so a dose matched on the centered error decides it -- but
# first the dose has to be found, and it is not the same number on the two models. This
# sweeps it at 16 bits, where the noise is the only error the cache carries.
#
# Read off the "per-channel" column at 16 bits (the axis does not matter for noise) and
# pick, for each model, the sigma whose centered error is nearest 0.35.
#
#   tmux new -s t4b -d "bash scripts/pod/t4b.sh 2>&1 | tee out/_t4b.log"
#   for f in out/kn_*.json; do python scripts/diag_compare.py $f --centered --bits 16; done

. "$(dirname "$0")/lib.sh"

D="--samples 8 --bits 16 --mask-ratio 0.5 --all-layers --logit-error"
MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B --model-type fast_dllm_v2"

for s in 0.02 0.05 0.10 0.20; do
  t=${s/./}
  run kn_fd_$t 24000   out/kn_fd_$t.json $PY20 scripts/check_key_error.py $MFD $D \
      --key-noise $s --dump out/kn_fd_$t.json
  run kn_20_$t $NEED20 out/kn_20_$t.json $PY20 scripts/check_key_error.py $M20 $D \
      --key-noise $s --dump out/kn_20_$t.json
done

echo "$(date +%H:%M) тезис 4: калибровка дозы шума — готово"
