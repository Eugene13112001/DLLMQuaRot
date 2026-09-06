"""What a stale cache costs a confidence-thresholded sampler, in steps.

Every damage number in this project is a share of committed decisions that
moved, and section 5 showed those decisions do not reach the answer: across
the whole grid -- ages 0 to 16, sixteen bits down to three -- the refresh
policy does not change a single GSM8K question, not the count and not which
ones. Nought to sixty-two percent of decisions changed, and identical sets of
solved problems.

So the price is not accuracy. The remaining candidate is throughput, and
until now it was measured as a proxy: the share of positions whose top-1
probability clears a threshold, computed off the same margin dumps the damage
came from. That is circular, and a reviewer would say so.

This measures it directly. The sampler stops taking a fixed number of tokens
per step and instead commits every position it is sure enough about, so the
number of steps a block needs becomes an observation rather than an input.
A cache that flattens confidence buys fewer commits per step and the block
takes longer -- in the units Fast-dLLM, dInfer and DMax quote.

The control is the exact configuration: sixteen bits refreshed every step,
where a block-causal prefix makes the cache lossless by construction. Any row
slower than that is paying for staleness or rounding and nothing else.

    bash scripts/llada2.sh scripts/check_parallelism.py \
        --model inclusionAI/LLaDA2.0-mini --model-type llada2_moe
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dllmquant.cache import BlockKVCache, KVCacheConfig  # noqa: E402
from dllmquant.config import DLLMQuantConfig, QuantConfig, TMASConfig  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402
from dllmquant.models import llada2_local, llada_local  # noqa: E402

from check_block_cache import text_ids  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="llada2_moe",
                    choices=["llada2_moe", "llada"],
                    help="the two families price staleness differently and "
                         "that is the comparison: on LLaDA2.0 a block-causal "
                         "prefix is exact so only the current block can age, "
                         "on LLaDA-1.5 attention is bidirectional and the "
                         "prefix ages too. The 1.75x measured on the MoE says "
                         "nothing about the dense model until this runs")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="the confidence a position must clear to commit. 0.9 "
                         "is what dInfer ships; the checkpoint's own generate "
                         "uses 0.95, and the proxy measurement was robust to "
                         "the choice, so this should be too")
    ap.add_argument("--bits", type=int, nargs="+", default=[16, 4, 3])
    ap.add_argument("--policies", nargs="+",
                    default=["every_n:1", "every_n:2", "every_n:4",
                             "every_n:8", "block"])
    args = ap.parse_args()

    cfg = DLLMQuantConfig(model_path=args.model, model_type=args.model_type,
                          dtype=args.dtype, device=args.device,
                          weight=QuantConfig(n_bits=16),
                          activation=QuantConfig(n_bits=16))
    adapter = build_adapter(cfg)
    adapter.load()
    print(adapter.describe())

    dense = args.model_type == "llada"
    cached_generate = (llada_local if dense else llada2_local).cached_generate

    n_layers = len(adapter.blocks)
    n_blocks = args.gen_length // args.block_length
    gen_cfg = TMASConfig(gen_length=args.gen_length,
                         block_length=args.block_length,
                         steps=args.gen_length)
    prompts = [text_ids(adapter, args.prompt_tokens, seed=i).unsqueeze(0)
               for i in range(args.samples)]

    print(f"\n{args.gen_length} tokens = {n_blocks} blocks of "
          f"{args.block_length}, {args.samples} prompts, commit above "
          f"{args.threshold}")
    print("steps per block is now an observation, not a schedule")

    def kv_config(bits: int, policy: str, every: int) -> KVCacheConfig:
        return KVCacheConfig(enabled=True, decoded_bits=bits,
                             masked_bits=bits, policy=policy,
                             refresh_every=every, key_axis="token")

    rows: List[Tuple[str, int, float, float, float]] = []
    for bits in args.bits:
        for spec in args.policies:
            name, _, interval = spec.partition(":")
            steps: List[int] = []
            for prompt in prompts:
                cache = BlockKVCache(
                    kv_config(bits, name, int(interval) if interval else 4),
                    n_layers)
                cached_generate(adapter, prompt, gen_cfg, cache,
                                reuse_window=True, threshold=args.threshold)
                steps.extend(cache.stats.steps_used)
            mean_steps = sum(steps) / max(len(steps), 1)
            # Tokens per step is the reciprocal in the units people quote, and
            # the full-refresh row is the speed a lossless cache reaches.
            rows.append((spec, bits, mean_steps,
                         args.block_length / max(mean_steps, 1e-9),
                         float(min(steps)) if steps else float("nan")))

    base = next((r[2] for r in rows if r[0] == "every_n:1" and r[1] == 16), None)
    print()
    print("=== steps a block needs, committing above the threshold ===")
    print(f"{'policy':>12} {'bits':>5} {'steps':>8} {'tok/step':>9} "
          f"{'best blk':>9} {'vs exact':>9}")
    print("-" * 56)
    for spec, bits, mean_steps, per_step, best in rows:
        slow = (mean_steps / base) if base else float("nan")
        print(f"{spec:>12} {bits:>5} {mean_steps:>8.2f} {per_step:>9.2f} "
              f"{best:>9.0f} {slow:>8.2f}x")

    print()
    print("  The control is 16 bits at every_n:1: a block-causal prefix makes")
    print("  that cache lossless by construction, so its step count is the")
    print("  speed the sampler reaches with nothing wrong. Every row above it")
    print("  is staleness or rounding paid for in time.")
    print()
    print("  Read against section 5, where the same grid does not change one")
    print("  GSM8K question at any width. If these rows separate, the damage")
    print("  is real and priced in throughput; if they do not, the cache is")
    print("  free on both counts and the margin tables describe something")
    print("  with no consequence at this canvas -- which would be worth")
    print("  saying just as plainly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
