# Point 1 on answers: does the price of cache bits depend on the refresh policy?
#
# Section 2.9 measured it on decisions (check_block_reuse, 1024 committed positions):
# 3 bits flip 5.6% of decisions with the current block refreshed every step and are
# indistinguishable from exact at every 4 steps. On answers the policy axis was never
# varied -- the flag did not reach the sampler (section 5) -- so this is its first run.
#
# --reuse-window is what makes the policy live on the MoE path. Checked in code before
# this queue was written: without it the current block is recomputed in full precision
# at every step (llada2_local.py, window="fresh") and --kv-policy touches nothing; with
# it the block's own K/V is stored quantized (cache.write_window) and read back stale
# between refreshes (window="reuse"). 'block' writes the window once per block and never
# again. The prefix is refreshed once per block either way, and under the block-causal
# mask that is exact, so every error below is the current block's.
#
# This regime is not what dInfer or Fast-dLLM ship (they never reuse the block being
# decoded): it is the upper bound of staleness on this model, and the regime where the
# two errors of section 2.7 actually meet.
#
# Two bits instead of four: four bits cost nothing on answers even in the shipped regime
# (181 = 181 of 200), so a four-bit row could only return zeros. Two bits cost 6 points
# with the prefix alone (84.5 vs 90.5, p = 0.017), which leaves room to watch the price
# shrink. Key axis is the per-channel scale everywhere (the default), so the QK-Norm
# collapse does not enter.
#
# Ordered so the first six cells answer the question: bits x {every step, every 4, block}.
#
#   tmux new -s pol -d "bash scripts/pod/policy.sh 2>&1 | tee out/_pol.log"

. "$(dirname "$0")/lib.sh"

C="--n-eval 200 --gen-length 512 --eval-steps 256 --kv-cache --reuse-window"
e() { n=$1; shift; run "$n" $NEED20 "out/$n.json" $EVAL20 $C "$@" --out "out/$n.json"; }

# control: at 16 bits, refreshed every step, the reused window is the exact one; this must
# reproduce the shipped 16-bit cache (90.50) answer for answer
e l20_P16_n1  --kv-bits 16 --kv-policy every_n --kv-refresh-every 1
e l20_P2_n1   --kv-bits 2  --kv-policy every_n --kv-refresh-every 1
e l20_P16_n4  --kv-bits 16 --kv-policy every_n --kv-refresh-every 4
e l20_P2_n4   --kv-bits 2  --kv-policy every_n --kv-refresh-every 4
e l20_P16_blk --kv-bits 16 --kv-policy block
e l20_P2_blk  --kv-bits 2  --kv-policy block

# second pass: the middle interval and a second width, to see the shape rather than two ends
e l20_P16_n2  --kv-bits 16 --kv-policy every_n --kv-refresh-every 2
e l20_P2_n2   --kv-bits 2  --kv-policy every_n --kv-refresh-every 2
e l20_P3_n1   --kv-bits 3  --kv-policy every_n --kv-refresh-every 1
e l20_P3_n4   --kv-bits 3  --kv-policy every_n --kv-refresh-every 4
e l20_P3_blk  --kv-bits 3  --kv-policy block

echo "$(date +%H:%M) политика x разрядность на ответах: готово"
