# How much harm the Jensen inflation actually does, measured instead of computed.
#
# obj.sh gave the term: 35.3% of the logit spread on Fast-dLLM-v2 at the dose we used,
# 12.0% on LLaDA2.0-mini. That number is an upper bound, not the effect. The closed form
# exp(Var/2) assumes the noised group self-averages -- many keys, each drawing its own
# noise, their sum tending to the expectation. Concentrated attention breaks that: a handful
# of keys carry the sum, and for a handful the inflation is a draw rather than its mean. On a
# toy in this model's regime the closed form said the prefix gains 13 points of mass and the
# realised shift was 1.4.
#
# So this measures what moved. --noise-prefix-only leaves the last 32 positions clean, the
# way generation recomputes the current block while the prefix carries the store's error, and
# the run prints the attention mass on those positions before and after the noise.
#
#   tmux new -s obj2 -d "bash scripts/pod/obj2.sh 2>&1 | tee out/_obj2.log"
#   for f in out/kj_fd_*.log out/kj_20_*.log; do echo "=== $f"; grep -A 7 "realised shift" $f; done
#
# Reading it: a shift of a point or two means the confound is small and the noise arm can be
# read again, with the caveat stated. Ten points or more and the arm stays withdrawn.
# Each model is run at its own dose for KL 0.22 and KL 0.47.

. "$(dirname "$0")/lib.sh"

D="--samples 16 --bits 16 --mask-ratio 0.5 --all-layers --logit-error --noise-prefix-only 32 --block-tail 32"
MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B --model-type fast_dllm_v2"

for s in 0.0393 0.0614; do
  t=${s/./}
  run kj_fd_$t 24000 out/kj_fd_$t.json $PY20 scripts/check_key_error.py $MFD $D \
      --key-noise $s --dump out/kj_fd_$t.json
done
for s in 0.1323 0.1961; do
  t=${s/./}
  run kj_20_$t $NEED20 out/kj_20_$t.json $PY20 scripts/check_key_error.py $M20 $D \
      --key-noise $s --dump out/kj_20_$t.json
done

echo "$(date +%H:%M) реализованный сдвиг массы внимания: готово"
