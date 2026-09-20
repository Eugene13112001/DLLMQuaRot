# Fast-dLLM-v2-7B: does the key axis decide the cache on a block dLLM without QK-Norm?
#
# This model has no norm on Q and K and a bias on k_proj, so whatever fixed-channel
# structure its keys carry cannot come from a norm gain. Two outcomes, both worth having:
# the axis still decides at two bits (then the cause is the bias, the mechanism reported
# for Qwen2.5 elsewhere, and our claim generalises to parameter-induced outliers), or it
# does not (then the claim is bounded to models with a spread-out QK-Norm gain).
#
# The checkpoint's own sampler runs: blocks of 32, sub-blocks of 8, commit at confidence
# 0.95, up to 2048 new tokens -- the harness of the BitSieve tables.
#
# fd_16 first and alone: it has to reproduce the published dense number (85.0, n = 200)
# before any quantized cell means anything. If it lands far from that, stop and find out
# why rather than reading the rest.
#
#   tmux new -s fd -d "bash scripts/pod/fd.sh 2>&1 | tee out/_fd.log"

. "$(dirname "$0")/lib.sh"

MFD="--model Efficient-Large-Model/Fast_dLLM_v2_7B"
NEEDFD=34000          # 7B in bf16 plus a 2048-token canvas and its cache
EVALFD="bash scripts/llada2.sh scripts/evaluate_fdv2.py $MFD"
C="--n-eval 200 --gen-length 2048 --threshold 0.95 --block-size 32 --small-block-size 8"
e() { n=$1; shift; run "$n" $NEEDFD "out/$n.json" $EVALFD $C "$@" --out "out/$n.json"; }

# control: lossless cache, the published setting
e fd_16       --kv-bits 16
if [ ! -s out/fd_16.json ]; then
  echo "$(date +%H:%M) !!! контрольная клетка не досчиталась, дальше не идём"
  exit 1
fi

# the axis at two bits, keys only changing
e fd_2_ch     --kv-bits 2
e fd_2_tok    --kv-bits 2 --kv-key-axis channel
e fd_2_quarot --kv-bits 2 --kv-key-axis channel --rotate-qk

# their operating point, for the record
e fd_4_ch     --kv-bits 4

echo "$(date +%H:%M) Fast-dLLM-v2: готово"
