# Do two layers of 28 carry the collapse?
#
# The per-layer profile of the attention movement (kl_profile.py on ke_afd) says the
# rotation's damage is not spread: at four bits QuaRot moves attention by KL 5.62 in layer
# 27 and 3.24 in layer 0, while the other 26 layers sit at 0.02, and 90% of layers are
# under 0.10. The model still scores 1.5. The mean movement therefore cannot be what
# accuracy follows -- three cells at the same mean score 1.5, 28.0 and 43.5 -- and the
# candidate is that the first and the last layer are worth more than the rest.
#
# Keeping two layers of 28 in full precision costs 7% of the cache, which is the price to
# state next to any rescue. BitSieve already keeps its first two layers dense, chosen
# without this measurement; if the edge layers are what matters, that choice has a reason
# and a cheaper form (the last layer matters more than the second).
#
#   tmux new -s edge -d "bash scripts/pod/edge.sh 2>&1 | tee out/_edge.log"
#
# Against: QuaRot 1.5, per token 0.0, per channel at two bits 18.0, reference 82.5.

. "$(dirname "$0")/lib.sh"

MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B"
EVALFD="bash scripts/llada2.sh scripts/evaluate_fdv2.py $MFD"
C="--n-eval 200 --gen-length 2048 --threshold 0.95 --block-size 32 --small-block-size 8"
e() { n=$1; shift; run "$n" 34000 "out/$n.json" $EVALFD $C "$@" --out "out/$n.json"; }

# The cell the profile points at: rotation, with the two loud layers left exact.
e fd_4_quarot_edge --kv-bits 4 --kv-key-axis channel --rotate-qk --kv-skip-layers 0 27
# The same two layers on the other schemes, to see whether this is about the rotation
# or about the layers.
e fd_4_tok_edge    --kv-bits 4 --kv-key-axis channel --kv-skip-layers 0 27
e fd_2_ch_edge     --kv-bits 2 --kv-skip-layers 0 27
# The control that matters: two arbitrary middle layers, same 7% of the cache. If this
# rescues as well, the rescue is the budget and not the layers.
e fd_4_quarot_mid  --kv-bits 4 --kv-key-axis channel --rotate-qk --kv-skip-layers 13 14

echo "$(date +%H:%M) крайние слои: готово"
