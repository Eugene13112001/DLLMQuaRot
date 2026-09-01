"""Do the scales have to travel with the cache?

A four-bit cache is not four bits. The scale is derived from the tensor being
stored, so it has to be stored with it: at a group of 128 that is 32 bits per
128 entries and the real width is 4.25, at a group of 32 it is 5.0. What the
tables in this project call 3.76x compression is 16/4.25, and the missing
quarter-bit is the scales.

A scale that was calibrated in advance lives in the model instead -- 80 KB for
this checkpoint against 32 GB of weights -- and then four bits are four bits
and the compression is 4.00x. The question is whether a scale calibrated on
one canvas is still right on the next one.

Two measured facts say it might be, and they are the reason this is the first
item of phase C rather than a guess:

* **The token axis won.** Four bits grouped along tokens were indistinguishable
  from an exact cache where the same four along channels were not, and the
  explanation was that K's outliers sit in *fixed channels*. A quantity that
  sits in a fixed channel is exactly the quantity that can be measured once.
* **The distribution moves with the mask ratio** -- outliers 4.1 against 8.7,
  decision margin 1.988 against 0.756 between a fully masked prefix and a
  decoded one. So a single static scale is the wrong object, and the fix is
  free: the mask ratio is known at every sampler step, so the scales are
  bucketed by it and picked by the current one.

The script runs in two parts, and the first can refute the second before it
costs anything.

**The survey** asks only how far a per-channel scale moves between canvases,
which needs one forward per canvas and no sweep at all. If a channel's scale
is the same number on every canvas, calibration will work; if it is not, no
table below can rescue it. It also reports whether the movement is *explained*
by the mask ratio, because that is what decides whether bucketing is a fix or
a decoration.

**The sweep** then measures the thing itself, and its rows are arranged so that
two different costs cannot be confused:

    K dynamic, token, group 128     the measured champion, 4.29 bits
    K dynamic, token, one group     the same granularity a static scale has,
                                    but taken from the canvas being stored --
                                    the ceiling any calibration is bounded by
    K static, one bucket            what a single calibration transfers
    K static, mask-ratio buckets    what bucketing recovers of the difference

The gap between rows 1 and 2 is the price of the coarser granularity; the gap
between 2 and 4 is the price of the scales being *stale* rather than *coarse*.
Reporting one number for both would leave a bad result uninterpretable.

    bash scripts/llada2.sh scripts/check_static_scales.py \
        --model inclusionAI/LLaDA2.0-mini --model-type llada2_moe --device cpu
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dllmquant.cache import (  # noqa: E402
    BlockKVCache,
    KVCacheConfig,
    StaticScaleBook,
    channel_scales,
)
from dllmquant.config import DLLMQuantConfig  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402
from dllmquant.models.llada2_local import (  # noqa: E402
    install_block_cache,
    logits_for_window,
    refresh_prefix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_block_cache import (  # noqa: E402
    Comparison,
    compare,
    decision_margin,
    full_logits,
    masked_canvas,
    pct,
    text_ids,
)


# --------------------------------------------------------------- accounting


def overhead_bits(
    axis: str, group: int, n_tokens: int, head_dim: int, scale_bits: int = 16
) -> float:
    """Bits per stored entry spent on the scales themselves.

    Counted from the actual shape rather than from the group size, because the
    two differ whenever the axis does not divide evenly: 224 tokens in groups
    of 128 is two groups, not 1.75, so the overhead is 32*2/224 = 0.286 bits
    and not the 0.250 the group size suggests. A quarter of a bit is small; the
    whole claim being tested is worth a quarter of a bit.
    """
    along, across = (n_tokens, head_dim) if axis == "token" else (head_dim, n_tokens)
    groups = math.ceil(along / min(group, along))
    return 2 * scale_bits * groups * across / (n_tokens * head_dim)


def pair_cost(k_bits: float, v_bits: float) -> Tuple[float, float]:
    """Mean width over K and V, and the compression against a 16-bit cache.

    K and V are the same size on this checkpoint (GQA reduces both alike), so
    the mean is the true average width of a cached number.
    """
    mean = (k_bits + v_bits) / 2
    return mean, 16.0 / mean


def warn_if_underpowered(measured: Dict[str, "Comparison"], samples: int,
                         commit_k: int, target_se: float = 0.03) -> None:
    """Say so when the table cannot resolve its own rows.

    `argmax@k` is a rate over ``samples * commit_k`` positions, so at four
    canvases it is sixteen of them: a resolution of 6.25 points and a standard
    error near seven. Two rows that differ by less than that differ by nothing,
    and the giveaway is an impossible ordering -- four bits scoring below
    three, which is what the first run of this sweep printed.

    Printed rather than left to the reader because this project has twice
    published a row that the sample size could not support.
    """
    rates = [c.agree_k for c in measured.values() if c.agree_k == c.agree_k]
    if len(rates) < 2:
        return
    n = max(samples * commit_k, 1)
    spread = max(rates) - min(rates)
    p = sum(rates) / len(rates)
    se = (p * (1 - p) / n) ** 0.5
    if se <= 0 or spread > 2 * se:
        # A standard error of zero means every row scored identically, which is
        # a degenerate table rather than an underpowered one -- the floor check
        # above it is what catches that.
        return
    # What it would take to resolve *this* spread at two standard errors, not
    # to reach some fixed error: an earlier version asked for a 3-point
    # standard error and so advised 17 canvases to a run that had already used
    # 24 and was still short.
    needed = (int(4 * p * (1 - p) / spread**2 / max(commit_k, 1)) + 1
              if spread > 0 else None)
    print(f"      UNDERPOWERED: argmax@k here is a rate over {n} positions "
          f"(standard error {100 * se:.1f} points) and the rows span "
          f"{100 * spread:.1f}. Nothing in this table is a difference.")
    if needed is None:
        print("      Every row scored alike, so no sample size resolves them: "
              "the configurations are equivalent at this width, or the table "
              "is degenerate and the floor row will say which.")
    else:
        print(f"      Resolving a {100 * spread:.1f}-point gap at two standard "
              f"errors needs about {needed} canvases against the {samples} "
              f"used here -- cost is linear in --samples. A gap this size may "
              f"also simply not exist.")


# ------------------------------------------------------------------ survey


def variance_split(groups: List[List[torch.Tensor]]) -> Tuple[float, float, float]:
    """How much of a channel's scale movement is the mask ratio, and how much
    is just the canvas.

    ``groups`` is one list of observations per mask-ratio bucket. Returns the
    typical canvas-to-canvas factor, the typical bucket-to-bucket factor, and
    the share of the variance the bucket explains.

    A one-way random-effects split rather than a ratio of spreads, because
    ``max/min`` is not comparable across different numbers of observations and
    the two axes never have the same number. Drawing 28 canvases from one
    distribution gives max/min 1.84 where 2 draws from the *same* distribution
    give 1.20 -- which is almost exactly the gap the first version of this
    survey reported and read as "the movement is not the mask ratio".

    Scales are multiplicative, so the split is taken in logs and reported back
    as factors.
    """
    per_group = [torch.stack(g).clamp(min=1e-12).log() for g in groups if len(g) > 1]
    if len(per_group) < 2:
        return float("nan"), float("nan"), float("nan")

    within = torch.stack([g.var(dim=0, unbiased=True) for g in per_group]).mean(0)
    means = torch.stack([g.mean(dim=0) for g in per_group])
    n = min(g.shape[0] for g in per_group)
    between = (means.var(dim=0, unbiased=True) - within / n).clamp(min=0.0)

    share = between / (between + within).clamp(min=1e-12)
    return (
        float(within.sqrt().exp().median()),
        float(between.sqrt().exp().median()),
        float(share.median()),
    )


def spread(values: List[torch.Tensor]) -> Tuple[float, float]:
    """How far one channel's scale moves across a list of observations.

    ``values`` is one flat tensor per observation, all the same length. Returns
    the median and the 90th percentile, over channels, of max/min -- a ratio
    rather than a difference because a scale is a multiplicative quantity and
    the channels differ from each other by more than any of them differs from
    itself.

    The bit width the scales were computed at cancels: a scale is a range
    divided by a fixed number of levels, so max/min over observations is the
    same figure at three bits and at four. The survey therefore says nothing
    about which width to use, and the sweep is not optional.
    """
    if len(values) < 2:
        return float("nan"), float("nan")
    stacked = torch.stack(values)                       # [obs, channels]
    ratio = stacked.amax(0) / stacked.amin(0).clamp(min=1e-12)
    return float(ratio.median()), float(ratio.quantile(0.9))


def top_channel_overlap(values: List[torch.Tensor], k: int = 8) -> Tuple[float, float]:
    """Do the same channels carry the outliers on every canvas?

    Mean pairwise overlap of the top-``k`` channels by scale, next to what
    overlap would be by chance. This is the claim underneath the token axis
    stated directly: if the fat channels are the same ones every time, a scale
    per channel is a property of the model, and if they are not, the axis
    result and this experiment are both accidents of one canvas.
    """
    if len(values) < 2:
        return float("nan"), float("nan")
    tops = [set(v.topk(min(k, v.numel())).indices.tolist()) for v in values]
    pairs = [
        len(a & b) / len(a | b)
        for i, a in enumerate(tops) for b in tops[i + 1:]
    ]
    n = values[0].numel()
    # Expected Jaccard of two independent k-subsets of n, to first order.
    chance = (k * k / n) / (2 * k - k * k / n)
    return sum(pairs) / len(pairs), chance


def run_survey(scales: Dict, ratios: List[float], sides: Tuple[str, ...]) -> None:
    """Print the cheap check: does a per-channel scale transfer at all?

    ``scales`` is keyed (mask_ratio, side, layer) -> list of one flat tensor per
    canvas.
    """
    layers = sorted({ell for _, _, ell in scales})
    print("\n=== survey: how far does a per-channel scale move? " + "=" * 26)
    print("spread = max/min of one channel's scale, median over channels [p90],")
    print("then median over layers. 1.00 would mean a scale is a constant of "
          "the model.")

    for side in sides:
        print(f"\n  {side}")
        print(f"    {'mask ratio':>12} {'across canvases':>18} "
              f"{'top-8 channels agree':>22}")

        per_ratio_median: Dict[float, Dict[int, torch.Tensor]] = {}
        for r in ratios:
            med, p90, overlaps, chance = [], [], [], float("nan")
            per_ratio_median[r] = {}
            for ell in layers:
                obs = scales.get((r, side, ell))
                if not obs:
                    continue
                m, p = spread(obs)
                med.append(m)
                p90.append(p)
                o, chance = top_channel_overlap(obs)
                overlaps.append(o)
                per_ratio_median[r][ell] = torch.stack(obs).median(0).values
            if not med:
                continue
            print(f"    {r:>12.2f} {_med(med):>10.2f} [{_med(p90):.2f}]"
                  f"{_med(overlaps):>16.2f} (chance {chance:.2f})")

        # And the split that the column above cannot give: max/min grows with
        # the number of observations, and the two axes never have the same
        # number, so comparing them directly reads a difference in sample size
        # as a difference in the world.
        within, between, share = [], [], []
        for ell in layers:
            groups = [scales[(r, side, ell)] for r in ratios
                      if scales.get((r, side, ell))]
            w, b, s = variance_split(groups)
            if w == w:
                within.append(w)
                between.append(b)
                share.append(s)
        if within:
            _verdict(_med(within), _med(between), _med(share), len(ratios))


def _med(xs: List[float]) -> float:
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float("nan")
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def _verdict(within: float, between: float, share: float, n_ratios: int) -> None:
    """State what the split implies, and refuse to state more than that.

    The two factors are comparable to each other -- they come from a variance
    decomposition, not from spreads over different numbers of draws -- and both
    are comparable to the thing being bought. A static scale saves the 0.25 to
    0.29 bits the dynamic one costs, so a scale that is off by a factor f wipes
    that out once log2(f) exceeds it: about x1.22.
    """
    if within != within or between != between:
        return
    import math

    print(f"      -> a channel's scale moves by x{within:.2f} from canvas to "
          f"canvas, and by x{between:.2f} from one mask-ratio bucket to the "
          f"next; the bucket explains {100 * share:.0f}% of the variance.")
    if n_ratios < 3:
        print(f"      (only {n_ratios} buckets: the between-bucket term is a "
              "crude estimate, and the honest reading is an upper bound on "
              "what bucketing could buy)")

    cost = math.log2(within) if within > 1 else 0.0
    print(f"      -> calibrating on other canvases misplaces the scale by "
          f"about {cost:.2f} bits of resolution, against the 0.25-0.29 bits a "
          f"dynamic scale costs to store.")
    if cost > 0.35:
        print("      -> so a static scale is arithmetically behind before the "
              "sweep runs: it saves a quarter-bit and loses more than that to "
              "a scale that no longer fits. Expect the sweep to say so, and "
              "report it as the negative result it is.")
    elif share > 0.5:
        print("      -> most of what does move is the mask ratio, which is "
              "exactly what a bucket removes and what a sampler knows for "
              "free.")
    else:
        print("      -> the movement is small enough to be worth the "
              "quarter-bit, and the mask ratio is not what drives it: a "
              "single calibration may be all that is needed.")
    print("      (a survey, not a result: it predicts the sweep, it does not "
          "replace it)")


def report_differences(measured: Dict[str, "Comparison"], group_size: int) -> None:
    """The three differences the rows exist to produce, stated as differences.

    A table of six numbers invites reading the best one; what the experiment
    actually decides is a subtraction, and each of the three has a separate
    consequence. Printed with the resolution of the metric attached, because
    `argmax@k` off a handful of committed positions has a coarse one and a gap
    inside it is not a gap.
    """
    champion = next((k for k in measured if k.startswith("dynamic, group")), None)
    ceiling = "dynamic, one group"
    single, bucketed = "static, 1 bucket", "static, buckets"

    def gap(a: str, b: str) -> Optional[Tuple[float, float, float]]:
        if a not in measured or b not in measured:
            return None
        x, y = measured[a], measured[b]
        se = (x.agree_k_se ** 2 + y.agree_k_se ** 2) ** 0.5
        return 100 * x.agree_k, 100 * y.agree_k, 100 * se

    lines = [
        ("granularity  ", champion, ceiling,
         "one scale per channel instead of one per 128 tokens"),
        ("transfer     ", ceiling, bucketed,
         "the same scales, calibrated elsewhere -- the experiment"),
        ("buckets      ", single, bucketed,
         "what the mask ratio was worth storing"),
    ]
    for name, a, b, why in lines:
        if a is None:
            continue
        g = gap(a, b)
        if g is None:
            continue
        lo, hi, se = g
        # Two standard errors, not one. One is a 68% interval, and calling that
        # "real" put this line in direct contradiction with the power check
        # printed three lines below it -- which uses two and said the same gap
        # was nothing.
        verdict = "real" if abs(hi - lo) > 2 * se else (
            "suggestive" if abs(hi - lo) > se else "within the resolution"
        )
        print(f"      {name} {lo:6.2f}% -> {hi:6.2f}%  "
              f"({hi - lo:+.2f} +- {se:.2f}, {verdict})  {why}")


# -------------------------------------------------------------- collection


class TapCache(BlockKVCache):
    """A 16-bit cache that hands the raw K/V to a collector on the way past.

    Subclassed rather than hooked so the tensors are the ones the cache would
    actually have stored -- after RoPE, after the prefix slice -- and not a
    second reconstruction of them that could drift from the first.
    """

    def __init__(self, n_layers: int, sink: Dict):
        super().__init__(
            KVCacheConfig(decoded_bits=16, masked_bits=16), n_layers
        )
        self.sink = sink

    def write(self, layer, k, v, mask=None):
        self.sink[layer] = (k.detach().float().cpu(), v.detach().float().cpu())
        return super().write(layer, k, v, mask)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="llada2_moe",
                    choices=["llada2_moe", "llada"],
                    help="the dense family is here to separate two outlier "
                         "stories that look like one. Its activations are far "
                         "more outlier-heavy than the MoE's -- ratio 244 "
                         "against 8.7 -- and yet its cache tolerates three "
                         "bits almost for free, 0.4%% against 5.6%%. If K's "
                         "channels are also stable there, the two are one "
                         "phenomenon and the inverse ordering needs "
                         "explaining; if they are not, the tensor that "
                         "decides for weights and the tensor that decides for "
                         "the cache are simply different, and the axis result "
                         "was never expected to transfer")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 3],
                    help="widths to compare static against dynamic at. Four "
                         "because that is the operating point the cache "
                         "tables settled on, three because a difference that "
                         "four bits absorb may still be there one bit down.")
    ap.add_argument("--mask-ratios", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0],
                    help="mask ratio of the cached prefix, and the quantity "
                         "the scales are bucketed by")
    ap.add_argument("--samples", type=int, default=4,
                    help="canvases the table is measured on")
    ap.add_argument("--calib-samples", type=int, default=4,
                    help="canvases the scales are calibrated on. Disjoint from "
                         "the measured ones by construction -- they are drawn "
                         "from different stretches of the corpus -- because a "
                         "scale calibrated on the canvas it is then tested on "
                         "is not a static scale, it is a dynamic one with "
                         "extra steps, and it is already in the table under "
                         "its own name.")
    ap.add_argument("--group-size", type=int, default=128,
                    help="group for the dynamic reference rows")
    ap.add_argument("--commit-k", type=int, default=4)
    ap.add_argument("--window-mask-ratio", type=float, default=1.0)
    ap.add_argument("--skip-survey", action="store_true")
    ap.add_argument("--survey-only", action="store_true",
                    help="stop after the cheap check. It costs one forward per "
                         "canvas and can refute the rest.")
    args = ap.parse_args()

    cfg = DLLMQuantConfig(
        model_path=args.model, model_type=args.model_type,
        dtype=args.dtype, device=args.device, device_map=args.device_map,
    )
    adapter = build_adapter(cfg)
    adapter.load()
    print(adapter.describe())

    model = adapter.model
    device = next(model.parameters()).device
    head_dim = adapter.head_dim
    n_layers = len(adapter.blocks)

    total = args.blocks * args.block_length
    prefix_len = total - args.block_length
    buckets = tuple(args.mask_ratios)

    print(f"\nsequence {total} tokens, prefix cached = {prefix_len}, "
          f"window = {args.block_length}")
    print(f"{args.samples} canvases measured, {args.calib_samples} calibrated on "
          f"(disjoint corpus stretches)")

    eval_texts = [text_ids(adapter, total, seed=i) for i in range(args.samples)]
    calib_texts = [text_ids(adapter, total, seed=1000 + i)
                   for i in range(args.calib_samples)]

    states = install_block_cache(model, BlockKVCache(KVCacheConfig(), n_layers))

    def canvas_for(text, seed, mask_ratio):
        return masked_canvas(adapter, text, args.block_length, prefix_len,
                             mask_ratio, args.window_mask_ratio,
                             seed=seed).to(device)

    def raw_kv(x, mask_ratio) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """One forward, keeping the exact K/V the cache would have written."""
        sink: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        tap = TapCache(n_layers, sink)
        tap.mask_ratio = mask_ratio
        for state in states:
            state.cache = tap
        refresh_prefix(model, states, x, prefix_len, args.block_length)
        return sink

    # ---- collect scales, build books ---------------------------------------

    sides = ("key", "value")
    survey: Dict[Tuple[float, str, int], List[torch.Tensor]] = {}
    books: Dict[Tuple[int, str, str], StaticScaleBook] = {}
    # bits -> calibrated on the whole trajectory, one bucket
    for bits in args.bits:
        for reduce in ("max", "mean"):
            books[(bits, "buckets", reduce)] = StaticScaleBook(
                bits=bits, buckets=buckets, reduce=reduce
            )
            books[(bits, "single", reduce)] = StaticScaleBook(
                bits=bits, buckets=(0.5,), reduce=reduce
            )

    print("\ncalibrating...")
    for mask_ratio in args.mask_ratios:
        for i, text in enumerate(calib_texts):
            x = canvas_for(text, 1000 + i, mask_ratio)
            sink = raw_kv(x, mask_ratio)
            for layer, (k, v) in sink.items():
                for side, t in (("key", k), ("value", v)):
                    for book in books.values():
                        book.observe(layer, side, t, mask_ratio)
                    if not args.skip_survey:
                        s, _ = channel_scales(t, args.bits[0])
                        survey.setdefault((mask_ratio, side, layer), []).append(
                            s.flatten()
                        )

    for book in books.values():
        book.freeze()
    print("  " + books[(args.bits[0], "buckets", "max")].describe())
    print("  a single bucket is the same size divided by "
          f"{len(buckets)}: the ablation that says whether the mask ratio "
          "was worth storing for")

    # ---- the survey, on the canvases that will be measured -----------------

    if not args.skip_survey:
        for mask_ratio in args.mask_ratios:
            for i, text in enumerate(eval_texts):
                x = canvas_for(text, i, mask_ratio)
                for layer, (k, v) in raw_kv(x, mask_ratio).items():
                    for side, t in (("key", k), ("value", v)):
                        s, _ = channel_scales(t, args.bits[0])
                        survey.setdefault((mask_ratio, side, layer), []).append(
                            s.flatten()
                        )
        run_survey(survey, list(args.mask_ratios), sides)
        if args.survey_only:
            return 0

    # ---- the sweep ---------------------------------------------------------

    canvas_cache: Dict[float, List] = {}

    def canvases(mask_ratio):
        if mask_ratio not in canvas_cache:
            batch = []
            for i, text in enumerate(eval_texts):
                x = canvas_for(text, i, mask_ratio)
                reference = full_logits(model, x, args.block_length)[:, prefix_len:]
                batch.append(
                    (x, reference, x[:, prefix_len:total] == adapter.mask_id)
                )
            canvas_cache[mask_ratio] = batch
        return canvas_cache[mask_ratio]

    def run_row(make_cache, batch, mask_ratio, scramble=False):
        pooled, rails = Comparison(), []
        for x, reference, committable in batch:
            cache = make_cache()
            cache.mask_ratio = mask_ratio
            for state in states:
                state.cache = cache
            refresh_prefix(model, states, x, prefix_len, args.block_length)
            if scramble:
                cache.scramble(torch.Generator().manual_seed(0))
            windowed = logits_for_window(
                model, states, x, prefix_len, total, args.block_length
            )
            pooled = pooled + compare(reference, windowed,
                                      committable=committable,
                                      top_k=args.commit_k)
            if cache.stats.clipped:
                rails.append(cache.stats.clip_rate)
        return pooled, (sum(rails) / len(rails) if rails else float("nan"))

    v_group = min(args.group_size, head_dim)
    v_overhead = overhead_bits("channel", v_group, prefix_len, head_dim)

    def dynamic_cache(bits, k_group):
        return lambda: BlockKVCache(
            KVCacheConfig(enabled=True, decoded_bits=bits, masked_bits=bits,
                          group_size=k_group, key_axis="token",
                          value_axis="channel"),
            n_layers,
        )

    def static_cache(bits, which, reduce, kinds=("key",)):
        book = books[(bits, which, reduce)]
        return lambda: BlockKVCache(
            KVCacheConfig(enabled=True, decoded_bits=bits, masked_bits=bits,
                          group_size=v_group, key_axis="token",
                          value_axis="channel",
                          scale_book=book, scale_book_kinds=tuple(kinds)),
            n_layers,
        )

    print("\n" + "=" * 78)
    print("V is held at dynamic, channel axis, group "
          f"{v_group} in every row but the last, so the K rows differ in one "
          "thing only.")
    print("eff.bits counts the scales the cache has to carry; a static scale "
          "is not counted because it lives in the model and does not grow "
          "with the sequence.")

    for bits in args.bits:
        k_dyn = bits + overhead_bits("token", min(args.group_size, prefix_len),
                                     prefix_len, head_dim)
        k_one = bits + overhead_bits("token", prefix_len, prefix_len, head_dim)
        v_dyn = bits + v_overhead

        rows = [
            ("dynamic, group %d" % min(args.group_size, prefix_len),
             dynamic_cache(bits, args.group_size), k_dyn, v_dyn,
             "   <-- the measured champion"),
            ("dynamic, one group",
             dynamic_cache(bits, prefix_len), k_one, v_dyn,
             "   <-- static granularity, this canvas's own numbers: the "
             "ceiling"),
            ("static, 1 bucket",
             static_cache(bits, "single", "max"), float(bits), v_dyn, ""),
            ("static, buckets",
             static_cache(bits, "buckets", "max"), float(bits), v_dyn, ""),
            ("static, buckets, mean",
             static_cache(bits, "buckets", "mean"), float(bits), v_dyn, ""),
            ("static K and V",
             static_cache(bits, "buckets", "max", ("key", "value")),
             float(bits), float(bits), "   <-- nothing beside the cache at all"),
        ]

        for mask_ratio in args.mask_ratios:
            batch = canvases(mask_ratio)
            print(f"\n--- {bits} bits, prefix mask ratio {mask_ratio:.2f} "
                  + "-" * 30)

            margins = [decision_margin(ref, com, args.commit_k)
                       for _, ref, com in batch]
            margin = sum(m for m, _ in margins) / len(margins)
            print(f"decision margin at the committed positions: {margin:.3f} "
                  "logits")
            print(f"{'K scales':>22} {'eff.bits':>9} {'compr':>6} "
                  f"{'rel. err':>10} {'vs champ':>12} {'argmax':>8} "
                  f"{'argmax@k':>8} {'':>5} {'railed':>7}")

            chance, _ = run_row(dynamic_cache(16, head_dim), batch, mask_ratio,
                                scramble=True)
            print(f"{'scrambled':>22} {'':>9} {'':>6} {chance.rel:>10.3e} "
                  f"{pct(chance.agree)} {pct(chance.agree_k)}"
                  f"      {'':>7}   <-- chance floor")

            control, _ = run_row(dynamic_cache(16, head_dim), batch, mask_ratio)
            print(f"{'16 bit (no-op)':>22} {16.0:>9.2f} {1.0:>6.2f} "
                  f"{control.rel:>10.3e} {pct(control.agree)} "
                  f"{pct(control.agree_k)}      {'':>7}   <-- windowing noise")

            measured: Dict[str, Comparison] = {}
            for label, make_cache, kb, vb, note in rows:
                c, railed = run_row(make_cache, batch, mask_ratio)
                measured[label] = c
                mean_bits, compr = pair_cost(kb, vb)
                se = c.agree_k_se
                se_txt = "     " if se != se else f"±{100 * se:3.0f}%"
                rail_txt = "      -" if railed != railed else f"{100 * railed:6.2f}%"
                # Paired against the champion row, canvas by canvas: each
                # row's own spread is mostly how hard the canvases were, and
                # that cancels in the difference.
                delta = "            "
                if measured and label != next(iter(measured)):
                    d, dse = c.paired_delta(measured[next(iter(measured))])
                    if d == d:
                        delta = f"{100 * d:+6.1f}±{100 * dse:4.1f}%"
                print(f"{label:>22} {mean_bits:>9.2f} {compr:>6.2f} "
                      f"{c.rel:>10.3e} {delta} {pct(c.agree)} {pct(c.agree_k)} "
                      f"{se_txt} {rail_txt}{note}")

            report_differences(measured, args.group_size)
            warn_if_underpowered(measured, args.samples, args.commit_k)

    print("\nRead the table as three differences, not six rows.\n"
          "  champion -> ceiling   what a per-channel scale costs in "
          "granularity, with nothing stale about it.\n"
          "  ceiling -> static     what calibrating on other canvases costs. "
          "This is the number the experiment exists for.\n"
          "  1 bucket -> buckets   what the mask ratio was worth. If it is "
          "zero, say so and drop the buckets: they are storage and a lookup.\n\n"
          "`railed` is the share of entries pushed to an end of the integer "
          "range. It is not zero for a perfect calibration -- clip_ratio puts "
          "the rails inside the observed range on purpose -- so read it "
          "against the ceiling row's granularity, and only as an explanation "
          "for a static row that is worse than it: a scale can also be wrong "
          "by being far too wide, and that failure does not appear here at "
          "all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
