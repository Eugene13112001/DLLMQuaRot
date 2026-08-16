"""What does reusing the block being decoded cost, along a whole trajectory?

Every cache number in this project was taken on **one window against a clean
prefix**. That made the measurements clean -- block-causal attention makes a
closed block exact, so staleness is structurally zero and the only error is
storage -- and it left the thesis half measured, because the two errors it is
about never met. It also left the largest question open: four bits can be free
for one step and not free for sixteen blocks in sequence, and no analytic table
can see that.

This runs the sampler along a full trajectory with the current block cached as
well, and compares the tokens it commits against the exact variant, block by
block.

Two things it measures that nothing else can:

* **What staleness costs at the level of committed tokens.** At 16 bits the
  storage is a no-op, so a difference from the exact run is staleness and
  nothing else. Rounding then enters as the bit width drops, and the two can be
  read against each other in the same table.
* **Whether error accumulates across blocks.** Attention across blocks is
  causal, so a block decoded from a stale cache writes K/V that every later
  block reads. If the per-block agreement falls with block index, it
  accumulates; if it is flat, each block pays only for itself.

**On the saving, honestly.** Reusing the current block's K/V does not remove a
forward -- the queries still have to be computed, because the sampler needs
logits for every masked position. What it removes is the K and V projections
for those positions, which is a fraction of one layer. The reason to build it
is that staleness lives here and nowhere else, not that it is fast. The table
reports refreshes avoided, which is what a real implementation would save, and
does not report seconds, which would be measuring this simulation rather than
the method.

    bash scripts/llada2.sh scripts/check_block_reuse.py \\
        --model inclusionAI/LLaDA2.0-mini --model-type llada2_moe --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dllmquant.cache import BlockKVCache, KVCacheConfig  # noqa: E402
from dllmquant.config import DLLMQuantConfig, TMASConfig  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402
from dllmquant.models.llada2_local import cached_generate  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_block_cache import text_ids  # noqa: E402


class ScrambledWindow(BlockKVCache):
    """Stores the current block with its entries shuffled along the token axis.

    The chance floor for a token-agreement metric. Two generations of one model
    agree on a great deal by prior alone, so a row that "kept 60%" says nothing
    until this row is beside it. Permutation rather than noise: the marginal
    distribution of every channel is preserved exactly, so the floor is not
    confounded by a scale the model never sees.
    """

    def write_window(self, layer, k, v, mask=None):
        kq, vq = super().write_window(layer, k, v, mask)
        for store, salt in ((self._wk, 0), (self._wv, 1)):
            t = store[layer]
            g = torch.Generator(device="cpu").manual_seed(2 * layer + salt)
            idx = torch.randperm(t.shape[-2], generator=g).to(t.device)
            store[layer] = t.index_select(-2, idx)
        return self._wk[layer], self._wv[layer]


def newly_committed(x: torch.Tensor, prev: torch.Tensor, lo: int, hi: int,
                    mask_id: int) -> List[int]:
    """Positions inside [lo, hi) that stopped being masked since ``prev``.

    The teacher-forced table is only as good as this: a position counts once,
    on the step it commits. An earlier version compared against a canvas of
    -1, so on the first call every position of the block read as newly
    committed -- the whole block was scored while still masked and then
    overwritten at step 0, which is why the first column printed 1/32 for
    every row including the control that must print 100%.
    """
    return [
        pos for pos in range(lo, min(hi, x.shape[-1]))
        if int(x[0, pos]) != mask_id and int(prev[0, pos]) == mask_id
    ]


def agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a == b).to(torch.float32).mean())


def per_block(a: torch.Tensor, b: torch.Tensor, bounds) -> List[float]:
    """Agreement inside each block, which is where accumulation would show."""
    return [agreement(a[0, lo:hi], b[0, lo:hi]) for lo, hi in bounds]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="llada2_moe", choices=["llada2_moe"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--gen-length", type=int, default=128,
                    help="tokens to generate. The point of this script is the "
                         "trajectory, so this is the axis that matters: at 128 "
                         "and a block of 32 it is four blocks in sequence.")
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--bits", type=int, nargs="+", default=[16, 4, 3],
                    help="16 first and not as a formality: there the storage "
                         "is a no-op, so whatever the row loses is staleness "
                         "with rounding held out.")
    ap.add_argument("--policies", nargs="+",
                    default=["block", "every_n:8", "every_n:4", "every_n:2",
                             "mask_ratio"],
                    help="refresh policy for the current block, as name or "
                         "name:interval. 'block' never refreshes inside a "
                         "block and is the cheapest and stalest thing "
                         "possible -- the upper bound on what reuse can cost.")
    ap.add_argument("--samples", type=int, default=4,
                    help="independent prompts, pooled. Token agreement is a "
                         "rate over gen_length positions per prompt, so this "
                         "is far better resolved than the window tables: at "
                         "128 tokens and 4 prompts it is 512 trials.")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--key-axis", default="token", choices=["channel", "token"])
    args = ap.parse_args()

    cfg = DLLMQuantConfig(
        model_path=args.model, model_type=args.model_type,
        dtype=args.dtype, device=args.device, device_map=args.device_map,
    )
    adapter = build_adapter(cfg)
    adapter.load()
    print(adapter.describe())

    n_layers = len(adapter.blocks)
    gen_cfg = TMASConfig(gen_length=args.gen_length,
                         block_length=args.block_length,
                         steps=args.gen_length)
    n_blocks = args.gen_length // args.block_length

    prompts = [text_ids(adapter, args.prompt_tokens, seed=i).unsqueeze(0)
               for i in range(args.samples)]

    def kv_config(bits: int, policy: str, every: int) -> KVCacheConfig:
        return KVCacheConfig(
            enabled=True, decoded_bits=bits, masked_bits=bits,
            group_size=args.group_size, key_axis=args.key_axis,
            value_axis="channel", policy=policy, refresh_every=every,
        )

    def run(bits: int, policy: str, every: int, reuse: bool,
            scramble: bool = False):
        outs, hits, ages = [], [], []
        cls = ScrambledWindow if scramble else BlockKVCache
        for prompt in prompts:
            cache = cls(kv_config(bits, policy, every), n_layers)
            out = cached_generate(adapter, prompt, gen_cfg, cache,
                                  reuse_window=reuse)
            outs.append(out)
            hits.append(cache.stats.window_hit_rate)
            ages.append(cache.stats.mean_window_age)
        return outs, (sum(hits) / len(hits)), (sum(ages) / len(ages))

    print(f"\ngenerating {args.gen_length} tokens = {n_blocks} blocks of "
          f"{args.block_length}, {args.samples} prompts pooled "
          f"({args.samples * args.gen_length} committed tokens per row)")

    # The reference: the exact variant. Storage is a no-op at 16 bits and the
    # current block is recomputed every step, so this is the dense sampler --
    # which is a tested identity, not an assumption.
    print("\nreference: current block recomputed every step, 16-bit storage "
          "(identical to the dense sampler by construction)")
    reference, _, _ = run(16, "block", 4, reuse=False)

    # Taken from the adapter rather than recomputed: it decides where the
    # blocks fall when the prompt does not divide evenly.
    _, bounds = adapter._sequence_layout(prompts[0].shape[1], gen_cfg)
    n_blocks = len(bounds)

    header = (f"{'policy':>12} {'bits':>5} {'hit rate':>9} {'age':>6} "
              f"{'tokens kept':>12}   per block")
    print("\n" + header)
    print("-" * len(header))

    # A floor for the token metric: the same reuse with the store shuffled
    # along the token axis. Two generations of one model agree on a great deal
    # by prior alone, so "kept 60%" means nothing without it.
    floor_outs, _, _ = run(16, "block", 4, reuse=True, scramble=True)

    for bits in args.bits:
        for spec in args.policies:
            name, _, interval = spec.partition(":")
            every = int(interval) if interval else 4
            outs, hit, age = run(bits, name, every, reuse=True)

            kept = sum(agreement(o[:, bounds[0][0]:], r[:, bounds[0][0]:])
                       for o, r in zip(outs, reference)) / len(outs)
            blocks = [sum(per_block(o, r, bounds)[i] for o, r in
                          zip(outs, reference)) / len(outs)
                      for i in range(n_blocks)]
            label = spec if len(spec) <= 12 else spec[:12]
            print(f"{label:>12} {bits:>5} {100 * hit:>8.1f}% {age:>6.1f} "
                  f"{100 * kept:>11.2f}%   "
                  + " ".join(f"{100 * b:5.1f}" for b in blocks))

    # ---- the same thing with the text held fixed --------------------------
    #
    # The table above cannot separate two very different things. Once a cached
    # run commits a different token, every later block is decoded from a
    # different text, so its disagreement is a fork, not further damage. Here
    # each commit is overwritten with the reference's token, so the two runs
    # read the same canvas at every step and what is counted is only what the
    # cache changed *at that step*. This is the accumulation measurement; the
    # one above is the end-to-end cost.

    def forced(bits: int, policy: str, every: int):
        hits = [[0, 0] for _ in bounds]
        for prompt, ref in zip(prompts, reference):
            cache = BlockKVCache(kv_config(bits, policy, every), n_layers)
            state = {"prev": None}

            def force(block_idx, step, lo, hi, logits, x, ref=ref, state=state):
                if state["prev"] is None:
                    # A canvas of masks, not of -1: with -1 every position of
                    # the first block reads as just-committed, so the whole
                    # block gets compared while still masked (1 of 32 agreeing
                    # by luck) and then overwritten wholesale at step 0. The
                    # first column was measuring that, and the blocks after it
                    # were measured against a block that had been corrected
                    # before it could be damaged.
                    state["prev"] = torch.full_like(x, adapter.mask_id)
                for pos in newly_committed(x, state["prev"], lo, hi,
                                          adapter.mask_id):
                    hits[block_idx][1] += 1
                    hits[block_idx][0] += int(x[0, pos] == ref[0, pos])
                    x[0, pos] = ref[0, pos]
                state["prev"] = x.clone()

            cached_generate(adapter, prompt, gen_cfg, cache,
                            reuse_window=True, on_step=force)
        return [h / max(n, 1) for h, n in hits]

    print("\n" + "=" * 78)
    print("the same rows with the text held to the reference: every commit is "
          "overwritten, so both runs read one canvas and the columns count "
          "only what the cache changed")
    print(f"\n{'policy':>12} {'bits':>5}   per block (committed tokens agreeing)")
    print("-" * 60)
    for bits in args.bits:
        for spec in args.policies:
            name, _, interval = spec.partition(":")
            blocks = forced(bits, name, int(interval) if interval else 4)
            print(f"{spec:>12} {bits:>5}   "
                  + " ".join(f"{100 * b:5.1f}" for b in blocks))
    print("\nFalling left to right here is accumulation and nothing else: the "
          "canvas is identical, so a later block that agrees less is reading a "
          "cache that earlier blocks damaged.")

    floor = sum(agreement(o[:, bounds[0][0]:], r[:, bounds[0][0]:])
                for o, r in zip(floor_outs, reference)) / len(floor_outs)
    print(f"\nchance floor (store shuffled along tokens): {100 * floor:.2f}% "
          "-- a row at this level carries nothing, whatever its hit rate")

    print("\n`same path` is a SENSITIVITY, not a quality. Any perturbation "
          "gives every step a small chance of committing a different token, "
          "and over a hundred-odd steps the trajectories part almost surely; "
          "everything after that differs for free. The row that proves it is "
          "4 bits with the block refreshed at every step -- no staleness at "
          "all, and the path still leaves. That same four-bit cache costs "
          "nothing on GSM8K (96.0% against 92.0%), so the destination is not "
          "worse, the route is different. Read the teacher-forced table for "
          "damage and this one for how easily the path moves.\n")
    print("Read the per-block columns left to right. Flat means each block "
          "pays only for its own staleness. Falling means the error "
          "accumulates: a block decoded from a stale cache writes K/V that "
          "every later block reads, and that is the one thing the window "
          "tables in this project cannot see.\n\n"
          "The 16-bit rows are staleness alone, since storage is a no-op "
          "there. The distance from them to the 4- and 3-bit rows is what "
          "rounding adds on top -- and whether the two add or one absorbs the "
          "other is the question this whole line of work opened with.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
