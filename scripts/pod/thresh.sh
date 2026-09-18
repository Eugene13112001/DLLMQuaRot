# Do the two results survive the decoder the BitSieve harness uses?
#
# Every LLaDA2.0-mini number so far ran the fixed schedule: 256 steps, two tokens committed
# per step. Fast-dLLM-v2 and BitSieve decode by confidence instead -- every masked position
# at or above 0.95 commits, at least one per step. The claims are within-model contrasts,
# so they should not depend on the sampler, but that is a prediction and this queue tests it
# on the cells that carry each claim. Sub-blocks of 8 are not reproduced: our sampler has
# none, so this isolates the commit rule and nothing else.
#
# Under a threshold a refresh interval counts steps, and a step can commit one token or many;
# the record carries mean_steps_per_block so the staleness can be read in tokens.
#
#   tmux new -s thr -d "bash scripts/pod/thresh.sh 2>&1 | tee out/_thr.log"

. "$(dirname "$0")/lib.sh"

C="--n-eval 200 --gen-length 512 --eval-steps 256 --kv-cache --threshold 0.95"
e() { n=$1; shift; run "$n" $NEED20 "out/$n.json" $EVAL20 $C "$@" --out "out/$n.json"; }

# reference: lossless cache, fresh current block
e l20_T16          --kv-bits 16

# claim 2, the key axis and the gain: 84.5 / 1.5 / 57.5 under the fixed schedule
e l20_T2_ch        --kv-bits 2
e l20_T2_tok       --kv-bits 2 --kv-key-axis channel
e l20_T2_tok_M     --kv-bits 2 --kv-key-axis channel --migrate-qk 1.0

# claim 1, staleness x width with the current block reused: price of 2 bits 7.0 at every
# step, 24.5 at every 2 under the fixed schedule. At 16 bits and every step the reused
# block is the fresh one (checked), so l20_T16 is that cell's reference.
e l20_T2_n1r       --kv-bits 2  --reuse-window --kv-policy every_n --kv-refresh-every 1
e l20_T16_n2r      --kv-bits 16 --reuse-window --kv-policy every_n --kv-refresh-every 2
e l20_T2_n2r       --kv-bits 2  --reuse-window --kv-policy every_n --kv-refresh-every 2

echo "$(date +%H:%M) порог 0.95: готово"
