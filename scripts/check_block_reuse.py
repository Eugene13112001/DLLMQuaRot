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
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dllmquant.cache import BlockKVCache, KVCacheConfig  # noqa: E402
from dllmquant.config import DLLMQuantConfig, TMASConfig  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402
from dllmquant.models import llada2_local, llada_local  # noqa: E402
from measure_drift import record_routes  # noqa: E402
from trace_window_path import force_routes  # noqa: E402

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
    ap.add_argument("--model-type", default="llada2_moe",
                    choices=["llada2_moe", "llada"],
                    help="'llada' is the dense family (LLaDA-1.5), where "
                         "the prefix is what ages: attention is "
                         "bidirectional, so a closed position keeps "
                         "attending to the masked tail and its K/V goes "
                         "stale on its own. Under llada2_moe's "
                         "block-causal mask the prefix is exact and only "
                         "the current block can be stale. Same two errors, "
                         "opposite regime -- which is the whole reason to "
                         "run both.")
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
    ap.add_argument("--pin-routes", action="store_true",
                    help="hold every router to the choice it made on the "
                         "all-masked canvas, for the whole sweep. On tensors "
                         "this already showed that a fifth to a third of what "
                         "the current block calls staleness is route churn "
                         "rather than the state moving, and that the router "
                         "turns a small arithmetic shift into a discrete "
                         "change of expert. Neither of those says what it does "
                         "to a decision, which is the quantity every table "
                         "here is in. Pinned, the rounding floor and the "
                         "staleness rows can be read again with the amplifier "
                         "switched off: if the floor collapses toward the "
                         "dense model's 0.7%% while the block row barely "
                         "moves, the router is amplifying rounding "
                         "selectively -- and if both move together it "
                         "amplifies whatever it is given")
    ap.add_argument("--rotate-qk", action="store_true",
                    help="insert QuaRot's R4 -- a head-wise Hadamard on Q and "
                         "K after RoPE, so the store holds rotated keys. It "
                         "predicts against two results here rather than for "
                         "them. The token axis wins because K's outliers sit "
                         "in fixed channels, and static scales work for the "
                         "same reason; R4 exists to smear exactly those "
                         "outliers across channels. So the axis gap should "
                         "narrow or vanish, and the rounding floor should "
                         "move. Run the 16-bit row first: attention is exactly "
                         "invariant under R4, so any change there is a bug, "
                         "not a finding.")
    ap.add_argument("--dump-margins", default=None,
                    help="write per-position decision margins to this JSON. "
                         "The teacher-forced canvas makes a position the same "
                         "position in every configuration, so the margins can "
                         "be joined across cells and the composition of the "
                         "two errors read off directly: the margin shift is a "
                         "linear functional of the perturbation, so the shifts "
                         "must add even where the lengths compose in "
                         "quadrature. Also records the top-1 probability, "
                         "which is what a confidence-thresholded sampler "
                         "commits on -- if quantization compresses it, the "
                         "achievable parallelism falls with it.")
    args = ap.parse_args()

    cfg = DLLMQuantConfig(
        model_path=args.model, model_type=args.model_type,
        dtype=args.dtype, device=args.device, device_map=args.device_map,
    )
    adapter = build_adapter(cfg)
    adapter.load()
    print(adapter.describe())

    if args.rotate_qk:
        from dllmquant.algos.quarot import install_qk_rotation
        install_qk_rotation(adapter)
        print("R4 installed: Q and K rotated head-wise after RoPE, so the "
              "store holds rotated keys")

    # Which sampler drives the run. The two share a signature so everything
    # below is written once, but they do not share a meaning for staleness:
    # see the --model-type help. `hit_rate` follows the same split -- on the
    # MoE path the measured quantity is the current block's store, on the
    # dense path there is no such store and the prefix itself is what gets
    # reused, so reading the window counters there would report a flat zero.
    dense = args.model_type == "llada"
    sampler = llada_local if dense else llada2_local
    cached_generate = sampler.cached_generate

    def hit_and_age(cache):
        if dense:
            ages = cache.stats.ages
            return cache.stats.hit_rate, (sum(ages) / len(ages) if ages else 0.0)
        return cache.stats.window_hit_rate, cache.stats.mean_window_age

    n_layers = len(adapter.blocks)
    gen_cfg = TMASConfig(gen_length=args.gen_length,
                         block_length=args.block_length,
                         steps=args.gen_length)
    n_blocks = args.gen_length // args.block_length

    prompts = [text_ids(adapter, args.prompt_tokens, seed=i).unsqueeze(0)
               for i in range(args.samples)]

    if args.pin_routes and dense:
        raise SystemExit("--pin-routes needs a router; the dense family has none")

    seq_len = args.prompt_tokens + args.gen_length

    def pin_for(prompt):
        """Hold this prompt's routers to what its own all-masked canvas chose.

        Per prompt, not once for the sweep. Recording on one prompt and
        forcing the result on the others is not "the amplifier off" but a
        different model: the tokens are sent to experts chosen for somebody
        else's text. The first version of this did exactly that, and it made
        every row two to three times worse instead of better -- which reads
        as a finding until one notices the reference row still says 100%,
        because both sides of that comparison were equally wrong.
        """
        if not args.pin_routes:
            return None
        device_ = next(adapter.model.parameters()).device
        x = torch.full((1, seq_len), adapter.mask_id, dtype=torch.long,
                       device=device_)
        x[:, :args.prompt_tokens] = prompt.to(device_)
        return force_routes(adapter.blocks, record_routes(adapter, x),
                            0, seq_len)

    if args.pin_routes:
        print(f"  routes pinned per prompt on its own all-masked canvas -- "
              f"every row below has the router's amplifier switched off")

    def kv_config(bits: int, policy: str, every: int) -> KVCacheConfig:
        return KVCacheConfig(
            enabled=True, decoded_bits=bits, masked_bits=bits,
            group_size=args.group_size, key_axis=args.key_axis,
            value_axis="channel", policy=policy, refresh_every=every,
        )

    def run(bits: int, policy: str, every: int, reuse: bool,
            scramble: bool = False, on_step=None):
        outs, hits, ages = [], [], []
        cls = ScrambledWindow if scramble else BlockKVCache
        for prompt in prompts:
            cache = cls(kv_config(bits, policy, every), n_layers)
            undo = pin_for(prompt)
            try:
                out = cached_generate(adapter, prompt, gen_cfg, cache,
                                      reuse_window=reuse, on_step=on_step)
            finally:
                if undo is not None:
                    undo()
            outs.append(out)
            hit, age = hit_and_age(cache)
            hits.append(hit)
            ages.append(age)
        return outs, (sum(hits) / len(hits)), (sum(ages) / len(ages))

    print(f"\ngenerating {args.gen_length} tokens = {n_blocks} blocks of "
          f"{args.block_length}, {args.samples} prompts pooled "
          f"({args.samples * args.gen_length} committed tokens per row)")

    # The reference: the exact variant. Storage is a no-op at 16 bits and the
    # current block is recomputed every step, so this is the dense sampler --
    # which is a tested identity, not an assumption.
    print("\nreference: current block recomputed every step, 16-bit storage "
          "(identical to the dense sampler by construction)")
    # The margin of each decision as it is taken, per block. Recorded on the
    # reference because it is a property of the trajectory, not of any cache.
    # It settles what the rising per-block columns mean: if the margin widens
    # with block index, later decisions are simply easier and any perturbation
    # would cost less there, which is a different claim from "the stale window
    # is diluted by a growing exact prefix".
    margins: List[List[float]] = [[] for _ in range(n_blocks)]
    margin_state = {"prev": None}

    def watch_margin(block_idx, step, lo, hi, logits, x):
        if margin_state["prev"] is None or (block_idx == 0 and step == 0):
            # Reset per prompt: the canvas restarts all-masked, and comparing
            # it against the previous prompt's finished one would find nothing
            # newly committed ever again.
            margin_state["prev"] = torch.full_like(x, adapter.mask_id)
        top2 = logits.float().topk(2, dim=-1).values
        for pos in newly_committed(x, margin_state["prev"], lo, hi,
                                   adapter.mask_id):
            gap = top2[0, pos - lo, 0] - top2[0, pos - lo, 1]
            margins[min(block_idx, n_blocks - 1)].append(float(gap))
        margin_state["prev"] = x.clone()

    reference, _, _ = run(16, "block", 4, reuse=False, on_step=watch_margin)

    # Taken from the adapter rather than recomputed: it decides where the
    # blocks fall when the prompt does not divide evenly.
    _, bounds = adapter._sequence_layout(prompts[0].shape[1], gen_cfg)
    n_blocks = len(bounds)

    header = (f"{'policy':>12} {'bits':>5} {'hit rate':>9} {'age':>6} "
              f"{'same path':>12}   per block")
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

    def forced(bits: int, policy: str, every: int, cls=BlockKVCache):
        """Returns per-block agreement and the exact set of flipped positions.

        The set is the point of the whole teacher-forced arrangement. With the
        canvas held identical across configurations, a position is comparable
        between runs, so the flips of one error can be intersected with the
        flips of another. If they largely coincide, sub-additivity has an
        explanation rather than just a measurement: both errors are knocking
        over the same near-ties, and buying down the smaller one changes
        nothing because those positions were already lost to the larger.
        """
        hits = [[0, 0] for _ in bounds]
        flipped = set()
        margins_here: Dict[str, List[float]] = {}
        for i, (prompt, ref) in enumerate(zip(prompts, reference)):
            cache = cls(kv_config(bits, policy, every), n_layers)
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
                lg = logits.float()
                top2v, top2i = lg.topk(2, dim=-1)
                probs = lg.softmax(dim=-1).amax(dim=-1)
                for pos in newly_committed(x, state["prev"], lo, hi,
                                          adapter.mask_id):
                    # Recorded at the step the position commits and before the
                    # reference overwrites it: after the overwrite the canvas
                    # is identical again, so this is the only moment where the
                    # cache's effect on this decision is visible.
                    #
                    # The margin is signed and taken *in the reference's
                    # ordering*: logit(reference token) minus the best of the
                    # rest. The cell's own top-2 gap is the wrong quantity for
                    # a regression across cells -- once a position flips, that
                    # gap belongs to a different pair of candidates, so the
                    # perturbed variable stops being the same variable as the
                    # reference's. That contamination is visible directly: the
                    # correlation against the reference margin is 0.738 at age
                    # 1 (5.4% flipped) and collapses to 0.198 at 'block' (62%
                    # flipped). The signed form has one ordering everywhere and
                    # crosses zero exactly when the token flips, so the flip
                    # set and the margin become the same measurement.
                    j = pos - lo
                    ref_tok = int(ref[0, pos])
                    best_i = int(top2i[0, j, 0])
                    if best_i == ref_tok:
                        signed = float(top2v[0, j, 0] - top2v[0, j, 1])
                    else:
                        signed = float(lg[0, j, ref_tok] - top2v[0, j, 0])
                    margins_here[f"{i}:{pos}"] = [
                        signed,
                        float(probs[0, j]),
                        # kept so the old, contaminated definition stays
                        # recoverable and the two can be compared on one dump
                        float(top2v[0, j, 0] - top2v[0, j, 1]),
                    ]
                    hits[block_idx][1] += 1
                    same = int(x[0, pos] == ref[0, pos])
                    hits[block_idx][0] += same
                    if not same:
                        flipped.add((i, pos))
                    x[0, pos] = ref[0, pos]
                state["prev"] = x.clone()

            undo = pin_for(prompt)
            try:
                cached_generate(adapter, prompt, gen_cfg, cache,
                                reuse_window=True, on_step=force)
            finally:
                if undo is not None:
                    undo()
        pooled = sum(h for h, _ in hits) / max(sum(n for _, n in hits), 1)
        return [h / max(n, 1) for h, n in hits], flipped, pooled, margins_here

    print("\n" + "=" * 78)
    print("the same rows with the text held to the reference: every commit is "
          "overwritten, so both runs read one canvas and the columns count "
          "only what the cache changed")
    print(f"\n{'policy':>12} {'bits':>5}   per block (committed tokens agreeing)")
    print("-" * 60)
    flips: Dict[Tuple[int, str], set] = {}
    frontier: Dict[Tuple[int, str], float] = {}
    margins_by_cell: Dict[str, Dict[str, List[float]]] = {}
    for bits in args.bits:
        for spec in args.policies:
            name, _, interval = spec.partition(":")
            blocks, flipped, pooled, cell_margins = forced(
                bits, name, int(interval) if interval else 4
            )
            margins_by_cell[f"{bits}/{spec}"] = cell_margins
            flips[(bits, spec)] = flipped
            frontier[(bits, spec)] = 1.0 - pooled
            print(f"{spec:>12} {bits:>5}   "
                  + " ".join(f"{100 * b:5.1f}" for b in blocks))

    # ---- does the best policy depend on the bit width? --------------------
    #
    # The question two neighbouring literatures leave open between them. The
    # refresh-policy papers tune when and where to recompute on an *exact*
    # cache; the cache-quantization papers pick a width and leave the policy
    # alone. Neither can see what happens to the first choice when the second
    # is made.
    #
    # The prediction is specific. At 16 bits the only error is staleness, so
    # refreshing more often helps all the way down. At 4 bits rounding puts a
    # floor under the damage, so past some interval a shorter one buys
    # nothing -- the knee moves, and a policy tuned without quantization
    # over-refreshes once the cache is quantized.

    if len(args.bits) > 1 and len(args.policies) > 1:
        print("\n" + "=" * 78)
        print("damage against refresh interval, one row per bit width")
        print("(share of committed tokens the cache changed, text held fixed)")
        header = "".join(f"{s:>12}" for s in args.policies)
        print(f"\n{'bits':>6}{header}")
        print("-" * (6 + 12 * len(args.policies)))
        for bits in args.bits:
            cells = "".join(
                f"{100 * frontier[(bits, s)]:>11.1f}%"
                if (bits, s) in frontier else f"{'-':>12}"
                for s in args.policies
            )
            print(f"{bits:>6}{cells}")

        print("\n      What to read: the *spacing* along each row, not the "
              "level. A row that keeps improving as the interval shortens is "
              "paying for staleness and nothing else. A row that flattens has "
              "hit its rounding floor, and every refresh past that point is "
              "spent on an error the bit width already fixed.")
        print("      If the rows flatten at different intervals, the best "
              "policy depends on the width -- which is the claim, and it says "
              "that policies tuned on an exact cache over-refresh a quantized "
              "one.")

    # The floor this table was missing. Rows near 95% do not need it; the row
    # that never refreshes inside a block sits low enough that "badly damaged"
    # and "carrying nothing" are not distinguishable without it.
    floor_blocks, _, _, _ = forced(16, "block", 4, cls=ScrambledWindow)
    print(f"{'scrambled':>12} {'--':>5}   "
          + " ".join(f"{100 * b:5.1f}" for b in floor_blocks)
          + "   <-- chance floor")
    if margins and any(margins):
        print(f"\n{'':>12} {'':>5}   decision margin per block, logits: "
              + " ".join(f"{sum(m) / len(m):5.2f}" if m else "  n/a"
                         for m in margins))
        print("      A margin that widens left to right means later decisions "
              "are simply harder to move, and the rising columns above would "
              "say little about the cache. A flat one leaves dilution as the "
              "explanation.")

    # ---- do the two errors flip the same positions? -----------------------
    #
    # The measurement behind sub-additivity, and the reason it is a mechanism
    # rather than a coincidence. Under teacher forcing the canvas is identical
    # across configurations, so a position is the same position in every run
    # and the flip sets can be intersected.

    low = min(args.bits)
    base = flips.get((low, "every_n:1"))
    if base is not None:
        print("\n" + "=" * 78)
        print(f"rounding alone ({low} bits, refreshed every step) flips "
              f"{len(base)} committed positions")
        print(f"\n{'staleness row':>16} {'flips':>6} {'shared':>8} "
              f"{'of smaller':>11} {'union':>7} {'together':>9}")
        for spec in args.policies:
            stale = flips.get((16, spec))
            both = flips.get((low, spec))
            if stale is None or both is None or not stale:
                continue
            shared = base & stale
            smaller = min(len(base), len(stale))
            print(f"{spec:>16} {len(stale):>6} {len(shared):>8} "
                  f"{100 * len(shared) / max(smaller, 1):>10.0f}% "
                  f"{len(base | stale):>7} {len(both):>9}")
        print("\n`together` against `union` is the whole question. Equal to "
              "the union means the two errors are independent and their "
              "damage adds; well below it means they are knocking over the "
              "same near-ties, and buying down the smaller error changes "
              "nothing because those positions were already lost to the "
              "larger. The second is what the allocation rule rests on: match "
              "the refresh interval to the bit width and stop paying for "
              "whichever error is not the binding one.")

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

    if args.dump_margins:
        # Written rather than analysed here on purpose: the regressions that
        # settle the composition need no GPU, and running them inside the
        # sweep would tie a five-line fit to a twenty-minute job.
        payload = {
            "config": {
                "model": args.model, "bits": args.bits,
                "policies": args.policies, "samples": args.samples,
                "gen_length": args.gen_length,
                "block_length": args.block_length,
                "group_size": args.group_size, "key_axis": args.key_axis,
                "prompt_tokens": args.prompt_tokens,
            },
            # "margin" is signed and in the reference's ordering:
            # logit(reference token) - max over the rest. Negative
            # exactly when the position flipped. "own_top2_gap" is the
            # older, contaminated definition, kept for comparison.
            "fields": ["margin", "top1_prob", "own_top2_gap"],
            "margin_def": "signed_reference_ordering",
            "cells": margins_by_cell,
        }
        with open(args.dump_margins, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        cells = len(margins_by_cell)
        positions = len(next(iter(margins_by_cell.values()), {}))
        print()
        print(f"wrote {cells} cells x ~{positions} positions to "
              f"{args.dump_margins}")
        print("What it is for, in three fits, all on joined positions.")
        print("  1. Additivity of the margin shift. Regress the shift of the "
              "both-errors")
        print("     cell on the sum of the two single-error shifts. The shift "
              "is a linear")
        print("     functional of the perturbation, so a slope of 1 is the "
              "prediction; a")
        print("     slope below 1 is systematic cancellation.")
        print("  2. Independence of sign. Correlate the two single-error "
              "shifts. Near zero")
        print("     means they compose by quadrature, which is why the flip "
              "count grows")
        print("     sub-additively even though the shifts themselves add.")
        print("  3. Shrinkage. Regress a cell's margin on the reference "
              "margin. The slope")
        print("     is the coefficient of 2608.06564 -- about 0.86 at four "
              "bits if their")
        print("     framework transfers to a diffusion model, and for "
              "staleness it is a")
        print("     number nobody has.")
        print("The top-1 probability is here for a fourth question: a "
              "confidence-thresholded")
        print("sampler commits every position above its threshold, so if "
              "quantization")
        print("compresses that probability it caps the parallelism the "
              "sampler can reach.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
