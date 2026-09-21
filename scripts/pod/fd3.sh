# Why does Fast-dLLM-v2 lose 64 points at two bits even with per-channel keys?
#
# fd_2_ch came out at 18.0 against 82.5 lossless, while BitSieve reports 74.0 at two bits
# on K and V for the same model (with selection). Before anything is read off the axis
# cells, the gap has to be located. Three suspects, one cell each:
#
#   which tensor -- keys alone at 2 bits, values alone at 2 bits;
#   the V group  -- this project scales V per token over the whole 128-channel head,
#                   BitSieve over groups of 32 channels;
#   the clipping -- this project keeps 95% of the min-max range, BitSieve all of it.
#
# fd_2_bs is the cache as BitSieve builds it (K per channel over the 32 tokens of a block,
# V per token over 32 channels, plain min-max). If it lands near 74, the collapse is ours
# to explain as a quantizer setting, not a property of the model.
#
#   tmux new -s fd3 -d "bash scripts/pod/fd3.sh 2>&1 | tee out/_fd3.log"

. "$(dirname "$0")/lib.sh"

MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B"
NEEDFD=34000
EVALFD="bash scripts/llada2.sh scripts/evaluate_fdv2.py $MFD"
C="--n-eval 200 --gen-length 2048 --threshold 0.95 --block-size 32 --small-block-size 8"
e() { n=$1; shift; run "$n" $NEEDFD "out/$n.json" $EVALFD $C "$@" --out "out/$n.json"; }

e fd_2_bs   --kv-bits 2 --kv-value-group-size 32 --kv-clip 1.0
e fd_2k     --kv-bits 16 --kv-key-bits 2
e fd_2v     --kv-bits 16 --kv-value-bits 2
e fd_2v_g32 --kv-bits 16 --kv-value-bits 2 --kv-value-group-size 32

echo "$(date +%H:%M) Fast-dLLM-v2, разбор двух бит: готово"
