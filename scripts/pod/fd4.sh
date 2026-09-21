# Fast-dLLM-v2 at four bits: where do the 8.5 points go?
#
# A dense four-bit prefix costs this model 8.5 points (82.5 -> 74.0), where LLaDA2.0-mini
# loses nothing and BitSieve, reading only the k = 64 selected entries, loses 0.5. At two
# bits every axis collapses, and those cells run for hours as the threshold sampler waits
# for confidence it never gets -- so the loss is taken apart here, at a width the model
# survives, where a cell takes forty minutes.
#
#   fd_4_bs    the cache as BitSieve builds it: V per token over 32 channels, plain min-max
#   fd_4k      keys alone at 4 bits
#   fd_4v      values alone at 4 bits, this project's whole-head V groups
#   fd_4v_g32  values alone at 4 bits, BitSieve's 32-channel V groups
#
#   tmux new -s fd4 -d "bash scripts/pod/fd4.sh 2>&1 | tee out/_fd4.log"

. "$(dirname "$0")/lib.sh"

MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B"
NEEDFD=34000
EVALFD="bash scripts/llada2.sh scripts/evaluate_fdv2.py $MFD"
C="--n-eval 200 --gen-length 2048 --threshold 0.95 --block-size 32 --small-block-size 8"
e() { n=$1; shift; run "$n" $NEEDFD "out/$n.json" $EVALFD $C "$@" --out "out/$n.json"; }

e fd_4_bs   --kv-bits 4 --kv-value-group-size 32 --kv-clip 1.0
e fd_4k     --kv-bits 16 --kv-key-bits 4
e fd_4v     --kv-bits 16 --kv-value-bits 4
e fd_4v_g32 --kv-bits 16 --kv-value-bits 4 --kv-value-group-size 32

echo "$(date +%H:%M) Fast-dLLM-v2, разбор четырёх бит: готово"
