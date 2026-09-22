# Fast-dLLM-v2: staleness x width with the checkpoint's own reuse of the current block.
#
# LLaDA2.0-mini showed the two errors compounding on answers under a fixed schedule and
# nearly additive under the confidence threshold (the price moving into steps). Fast-dLLM-v2
# has its own switch for reusing the block being decoded (use_block_cache): full at each
# sub-block start, then only the sub-block recomputed and the rest read back stale. With the
# cache quantized, the stale entries are rounded too. The bias is kept out of the store in
# every quantized cell, so what is measured is width and staleness, not the bias outlier.
#
# Reference cells without block reuse come from fd6/fd7: fd_16 (82.5), fd_4_ch_pb (85.0),
# fd_3_ch_pb.
#
#   tmux new -s fd8 -d "bash scripts/pod/fd8.sh 2>&1 | tee out/_fd8.log"

. "$(dirname "$0")/lib.sh"

MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B"
NEEDFD=34000
EVALFD="bash scripts/llada2.sh scripts/evaluate_fdv2.py $MFD"
C="--n-eval 200 --gen-length 2048 --threshold 0.95 --block-size 32 --small-block-size 8"
e() { n=$1; shift; run "$n" $NEEDFD "out/$n.json" $EVALFD $C "$@" --out "out/$n.json"; }

e fd_bc16     --kv-bits 16 --use-block-cache
e fd_bc4_pb   --kv-bits 4 --pre-bias --use-block-cache
e fd_bc3_pb   --kv-bits 3 --pre-bias --use-block-cache

echo "$(date +%H:%M) Fast-dLLM-v2, устаревание x разрядность: готово"
