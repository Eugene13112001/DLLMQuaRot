# Why is the quantizer's error gentler than noise of the same movement?
#
# On Fast-dLLM-v2 independent noise at KL 0.22 scores 3.0, where its own quantizers near
# that movement score 43.5 (KL 0.348) and 58.0 (KL 0.117). The sharpest hypothesis is
# correlation: a per-channel scale rounds every position of a channel the same way and a
# bias displaces them identically, while the noise so far drew a number per entry. A
# displacement common to all keys contributes one constant per query, which softmax reads
# against its own mean and largely ignores; an independent one reorders the keys.
#
# --kv-key-noise-mode shared is that correlation with nothing else of the quantizer in it:
# one vector per channel per write, added to every position. This sweeps its dose, because
# it is not the iid dose -- on a toy tensor the same size moves attention less than a fifth
# as far, so a much larger sigma is needed for the same movement, and how much larger is
# itself the answer to how much of the gentleness is correlation.
#
#   tmux new -s t4e -d "bash scripts/pod/t4e.sh 2>&1 | tee out/_t4e.log"
#   python scripts/noise_dose.py out/ks_fd_*.json --target-kl 0.22
#
# Then one evaluation at the matched movement, against iid's 3.0 and the reference 82.5:
#   shared at KL 0.22 answers near the reference -> the gentleness is correlation, and the
#     quantizer is survivable because its error is shared across positions, not because of
#     anything else about it.
#   shared at KL 0.22 also dies -> correlation is not it, and what is left is the path the
#     error takes through generation rather than its shape on a probe canvas.

. "$(dirname "$0")/lib.sh"

D="--samples 8 --bits 16 --mask-ratio 0.5 --all-layers --logit-error --key-noise-mode shared"
MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B --model-type fast_dllm_v2"

for s in 0.05 0.10 0.20 0.40; do
  t=${s/./}
  run ks_fd_$t 24000 out/ks_fd_$t.json $PY20 scripts/check_key_error.py $MFD $D \
      --key-noise $s --dump out/ks_fd_$t.json
done

echo "$(date +%H:%M) скоррелированный шум: калибровка дозы — готово"
