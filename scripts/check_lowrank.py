"""Is what quantization throws away worth correcting, or worth a bit?

`K ~ Q3(K) + A.B`. On weights the idea is established -- LQER, CALDERA,
ZeroQuant-V2 -- and on a diffusion LM's cache nobody has tried it, DART
included. The hypothesis in the plan was specific: rank 8 costs 10% of memory,
so three bits plus a rank-8 correction is about 3.3 effective bits against 4
flat.

**That arithmetic counts numbers, not bits, and it is off by a factor of
four.** A 224x128 prefix is 28672 entries and a rank-8 correction is 8*(224 +
128) = 2816 numbers, which is indeed 10% -- but the entries are three bits and
the numbers are fp16. In bits the correction costs 1.57 per entry, so "3 bits +
rank 8" is 4.57 effective and it is already losing to four bits flat (4.29)
before anything is measured. It reaches 3.3 only if the factors are themselves
quantized to four bits, which has an error of its own. So `--factor-bits` is a
swept axis here rather than a constant, and every row is priced in bits.

The survey runs first, because it can close the question without a sweep. It
needs one forward per canvas and reports, per layer and per side:

* how much of the residual's energy a rank-r correction captures, and
* what is left of the residual afterwards, **against what one more flat bit
  would have left** at a comparable or lower total width.

The second is the comparison that matters, and it is the one the plan's
mandatory control names: the same bits spent on a finer group -- or, cheaper
still, spent flat. A correction that recovers half the residual has not earned
anything if half a bit would have recovered more.

There is a reason to expect it to recover a great deal and still lose. Grouping
along tokens gives every channel its own scale, so a channel four times wider
rounds four times coarser and the residual inherits the tensor's channel
structure: a few outlier channels carry most of its energy, and a low-rank
factorization finds them. But finding them is not the same as being the
cheapest way to fix them -- spending the bits directly on those channels is
mixed precision, and it does not need a rank at all.

    bash scripts/llada2.sh scripts/check_lowrank.py \
        --model inclusionAI/LLaDA2.0-mini --model-type llada2_moe \
        --device cpu --survey-only
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dllmquant.cache import (  # noqa: E402
    BlockKVCache,
    KVCacheConfig,
    channel_range,
    lowrank_bits,
    lowrank_residual,
    mixed_precision,
    quantize_kv,
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
    full_logits,
    masked_canvas,
    pct,
    text_ids,
)
from check_static_scales import (  # noqa: E402
    TapCache,
    overhead_bits,
    pair_cost,
    warn_if_underpowered,
)


def rms(x: torch.Tensor) -> float:
    return float(x.float().pow(2).mean().sqrt())


def survey_side(
    tensors: List[torch.Tensor],
    bits: int,
    ranks: List[int],
    axis: str,
    group: int,
    factor_bits: int,
    mixed: List[Tuple[int, int]] = (),
    factor_scales: str = "column",
) -> List[Tuple]:
    """Residual energy recovered per rank, priced against one more flat bit.

    Returns one row per rank plus the flat-bit reference, all as fractions of
    the uncorrected residual, so the two are directly comparable and neither
    depends on the scale of the tensor.

    Each rank row also carries the comparison stated as a single number. One
    extra flat bit multiplies the residual by some factor `f`, and that is the
    local exchange rate between bits and error; a correction that multiplies it
    by `g` is therefore worth `log g / log f` flat bits. Divide by what it
    actually cost and the row says outright whether the rank beat the bit. The
    subtraction was being done by hand off the printed table, which is where
    arithmetic errors come from.
    """
    rows = []
    n_tokens, head_dim = tensors[0].shape[-2], tensors[0].shape[-1]

    base, capt, left = [], {r: [] for r in ranks}, {r: [] for r in ranks}
    mixed_left = {m: [] for m in mixed}
    finer = []
    for t in tensors:
        q = quantize_kv(t, bits, group, axis=axis)
        b = rms(t - q)
        base.append(b)
        for r in ranks:
            corr, c = lowrank_residual(t, q, r, factor_bits, factor_scales)
            capt[r].append(c)
            left[r].append(rms(t - q - corr) / max(b, 1e-12))
        for n_ch, extra in mixed:
            wide = mixed_precision(t, bits, extra, n_ch, group, axis=axis)
            mixed_left[(n_ch, extra)].append(rms(t - wide) / max(b, 1e-12))
        finer.append(rms(t - quantize_kv(t, bits + 1, group, axis=axis)) / max(b, 1e-12))

    def mean(xs):
        return sum(xs) / len(xs)

    flat_over = overhead_bits(axis, group, n_tokens, head_dim)
    flat_left = mean(finer)
    rate = math.log(flat_left) if 0.0 < flat_left < 1.0 else float("nan")

    def priced(name, captured, left_x, cost):
        worth = (math.log(left_x) / rate
                 if rate == rate and left_x > 0 else float("nan"))
        return (name, captured, left_x, bits + flat_over + cost, worth,
                worth / cost if cost else float("nan"))

    for r in ranks:
        rows.append(priced(
            f"rank {r}", mean(capt[r]), mean(left[r]),
            lowrank_bits(r, n_tokens, head_dim, factor_bits, factor_scales=factor_scales),
        ))
    for n_ch, extra in mixed:
        rows.append(priced(
            f"{n_ch}ch +{extra}b", float("nan"), mean(mixed_left[(n_ch, extra)]),
            extra * min(n_ch, head_dim) / head_dim,
        ))
    rows.append(("+1 bit flat", float("nan"), flat_left, bits + 1 + flat_over,
                 1.0, 1.0))
    return rows


def residual_geometry(
    tensors: List[torch.Tensor], bits: int, axis: str, group: int
) -> Dict[str, float]:
    """Along *what* does the residual's dominant direction run?

    A rank-1 component of a [T, head_dim] residual is an outer product, and it
    can be concentrated on either side. If its channel factor is one channel,
    the structure is an outlier *channel* and the cheaper fix is precision
    spent there. If its token factor is one position, the structure is an
    outlier *position* -- an attention sink -- and this stops being the
    low-rank item of phase C and becomes the sink item.

    The measure on each side is the participation ratio of the top singular
    vector: `1 / sum(w^4)` for a unit vector `w`, which reads as "how many
    components carry this direction". One means a single channel or a single
    position; head_dim means the direction is spread over everything and the
    correction is genuinely a rotation rather than a spike.

    Also reported: the modal position that dominates, because a sink at 0 and a
    sink that wanders are different findings -- and the second is the one this
    project expects (arXiv 2510.15731).
    """
    tok_eff, chan_eff, top_share, top_idx, idx_share = [], [], [], [], []
    widest = []
    for t in tensors:
        lo, hi = channel_range(t)
        widest.append((hi - lo)[:, 0, :].topk(4, dim=-1).indices)  # [heads, 4]
        q = quantize_kv(t, bits, group, axis=axis)
        resid = (t.float() - q.float())
        u, s, vh = torch.linalg.svd(resid, full_matrices=False)

        energy = s.pow(2)
        top_share.append(float((energy[..., 0] / energy.sum(-1).clamp(min=1e-12)).mean()))

        left = u[..., :, 0].abs()                       # [B, heads, T]
        right = vh[..., 0, :].abs()                     # [B, heads, head_dim]
        for vec, sink in ((left, tok_eff), (right, chan_eff)):
            w = vec.pow(2)
            w = w / w.sum(-1, keepdim=True).clamp(min=1e-12)
            sink.append(float((1.0 / w.pow(2).sum(-1)).mean()))

        best = left.argmax(-1).flatten()
        top_idx.extend(best.tolist())
        share = left.pow(2) / left.pow(2).sum(-1, keepdim=True).clamp(min=1e-12)
        idx_share.append(float(share.amax(-1).mean()))

    modal = max(set(top_idx), key=top_idx.count)

    # Do the widest channels stay the same ones from canvas to canvas? Mixed
    # precision needs a fixed index list, and if the list moved it would be an
    # oracle here and unbuildable in deployment. Mean pairwise overlap of each
    # head's top four, against 4/128 by chance.
    agree = []
    for i, a in enumerate(widest):
        for b in widest[i + 1:]:
            for h in range(a.shape[0]):
                sa, sb = set(a[h].tolist()), set(b[h].tolist())
                agree.append(len(sa & sb) / len(sa | sb))

    return {
        "widest_agreement": sum(agree) / len(agree) if agree else float("nan"),
        "top_share": sum(top_share) / len(top_share),
        "tokens": sum(tok_eff) / len(tok_eff),
        "channels": sum(chan_eff) / len(chan_eff),
        "modal_position": modal,
        "modal_agreement": top_idx.count(modal) / len(top_idx),
        "position_share": sum(idx_share) / len(idx_share),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="llada2_moe", choices=["llada2_moe"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--bits", type=int, default=3,
                    help="base width the correction sits on top of. Three, "
                         "because four are already free on the token axis and "
                         "there is nothing there for a correction to recover.")
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--factor-bits", type=int, nargs="+", default=[16, 4],
                    help="precision of A and B. 16 is what LQER stores; 4 is "
                         "what the plan's 3.3-bit figure silently assumed.")
    ap.add_argument("--factor-scales", default="column",
                    choices=["column", "token"],
                    help="how finely A's own scales are cut. 'token' is one "
                         "per token, which at rank 1 costs five times the "
                         "factor it describes -- the first survey charged "
                         "that and was pricing scales, not the correction. "
                         "'column' is one per rank component.")
    ap.add_argument("--mixed", nargs="*", default=["1x4", "2x4", "4x4", "2x8"],
                    help="mixed-precision controls, as CHANNELSxEXTRABITS: "
                         "give the widest N channels of each head that many "
                         "more bits. This is the control the geometry asked "
                         "for -- the residual's dominant direction is one "
                         "channel, and a rank-1 term pays for an entire outer "
                         "product to encode it. Empty to skip.")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--key-axis", default="token", choices=["channel", "token"])
    ap.add_argument("--mask-ratios", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--commit-k", type=int, default=4)
    ap.add_argument("--window-mask-ratio", type=float, default=1.0)
    ap.add_argument("--survey-only", action="store_true",
                    help="stop after the residual spectrum. It costs one "
                         "forward per canvas and can close the question.")
    ap.add_argument("--skip-survey", action="store_true",
                    help="go straight to the sweep. The survey collects raw K "
                         "for every canvas at every mask ratio, so once its "
                         "answer is known it is the more expensive half.")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="layers to print in the survey; default is first, "
                         "middle and last")
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
    texts = [text_ids(adapter, total, seed=i) for i in range(args.samples)]
    states = install_block_cache(model, BlockKVCache(KVCacheConfig(), n_layers))

    def canvas_for(i, mask_ratio):
        return masked_canvas(adapter, texts[i], args.block_length, prefix_len,
                             mask_ratio, args.window_mask_ratio,
                             seed=i).to(device)

    def raw_kv(x, mask_ratio):
        sink: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        tap = TapCache(n_layers, sink)
        tap.mask_ratio = mask_ratio
        for state in states:
            state.cache = tap
        refresh_prefix(model, states, x, prefix_len, args.block_length)
        return sink

    layers = args.layers or sorted({0, n_layers // 2, n_layers - 1})
    mixed = []
    for spec in args.mixed or ():
        n_ch, extra = (int(p) for p in spec.lower().split("x"))
        mixed.append((n_ch, extra))

    # ---- survey ------------------------------------------------------------

    if args.skip_survey:
        # Worth skipping once the geometry is known: the survey collects raw K
        # for every canvas at every mask ratio and factorizes it, which at two
        # dozen canvases is more forwards than the sweep itself.
        print("\nsurvey skipped")
    else:
        print(f"\n=== residual spectrum, {args.bits} bits, K along "
              f"{args.key_axis}s, group {args.group_size} " + "=" * 12)
        print("captured = share of the residual's energy the correction recovers")
        print("left     = what remains, as a fraction of the uncorrected residual")
        print("bits     = total width per entry, correction and scales included")
        print("worth    = the same error reduction, expressed in flat bits at the "
              "exchange rate the control row measures")
        print("x        = worth / cost. Above 1 the rank beat the bit; below it, "
              "the bits belonged in the base width.")

    for mask_ratio in (() if args.skip_survey else args.mask_ratios):
        collected: Dict[int, List[torch.Tensor]] = {}
        for i in range(args.samples):
            for layer, (k, _) in raw_kv(canvas_for(i, mask_ratio), mask_ratio).items():
                if layer in layers:
                    collected.setdefault(layer, []).append(k)

        print(f"\n--- prefix mask ratio {mask_ratio:.2f} " + "-" * 37)

        # Where the dominant direction lives, before anything is priced. A
        # correction that wins without this is a win with no mechanism, and a
        # spike on one axis is a cheaper fix than a rank on both.
        print(f"    {'layer':>5} {'top dir':>8} {'tokens':>7} {'channels':>9} "
              f"{'pos':>5} {'agree':>6} {'widest':>7}")
        for layer in sorted(collected):
            g = residual_geometry(collected[layer], args.bits, args.key_axis,
                                  args.group_size)
            note = ""
            if g["tokens"] < 4:
                note = "   <-- a few positions: this is a sink, not a rotation"
            elif g["channels"] < 1.5:
                note = "   <-- one channel: widening it should beat the rank"
            elif g["channels"] < 8:
                note = "   <-- spread over several: the rank should win"
            print(f"    {layer:>5} {100 * g['top_share']:>7.1f}% "
                  f"{g['tokens']:>7.1f} {g['channels']:>9.1f} "
                  f"{g['modal_position']:>5} {100 * g['modal_agreement']:>5.0f}% "
                  f"{100 * g['widest_agreement']:>6.0f}%{note}")
        print("    top dir = share of residual energy in the first singular "
              "direction; tokens/channels = how many components carry it; "
              "pos/agree = the position that dominates it most often and how "
              "often (chance is 1/heads); widest = do the widest four channels "
              "stay the same ones across canvases, which is what mixed "
              "precision needs and what makes its index list free")

        for factor_bits in args.factor_bits:
            print(f"\n  factors at {factor_bits} bits")
            print(f"    {'layer':>5} {'correction':>12} {'captured':>9} "
                  f"{'left':>7} {'bits':>7} {'worth':>6} {'cost':>6} {'x':>6}")
            base_bits = args.bits + overhead_bits(
                args.key_axis, args.group_size,
                collected[sorted(collected)[0]][0].shape[-2], head_dim,
            )
            for layer in sorted(collected):
                rows = survey_side(collected[layer], args.bits, args.ranks,
                                   args.key_axis, args.group_size, factor_bits,
                                   mixed=mixed, factor_scales=args.factor_scales)
                best = max((r[5] for r in rows if r[0] != "+1 bit flat"),
                           default=float("nan"))
                for name, captured, left, bits, worth, ratio in rows:
                    cap = "      -" if captured != captured else f"{100 * captured:6.1f}%"
                    if name == "+1 bit flat":
                        note, cost = "   <-- the control", 1.0
                    else:
                        cost = bits - base_bits
                        note = "   <-- beats the bit" if ratio > 1.15 else ""
                        if ratio == best and ratio > 1.15:
                            note += ", best here"
                    print(f"    {layer:>5} {name:>12} {cap:>9} "
                          f"{left:>7.3f} {bits:>7.2f} {worth:>6.2f} "
                          f"{cost:>6.2f} {ratio:>6.2f}{note}")
                print()

    if args.survey_only:
        print("Read it as a rate: a correction is worth its bits only if it "
              "leaves less residual than the flat row does at no more total "
              "width. If it does not, the finding is negative and complete -- "
              "and a negative here is a result, because the technique is "
              "established on weights and untried on this kind of cache.")
        return 0

    # ---- sweep -------------------------------------------------------------

    canvas_cache: Dict[float, List] = {}

    def canvases(mask_ratio):
        if mask_ratio not in canvas_cache:
            batch = []
            for i in range(args.samples):
                x = canvas_for(i, mask_ratio)
                reference = full_logits(model, x, args.block_length)[:, prefix_len:]
                batch.append((x, reference, x[:, prefix_len:total] == adapter.mask_id))
            canvas_cache[mask_ratio] = batch
        return canvas_cache[mask_ratio]

    def run_row(make_cache, batch, mask_ratio, scramble: bool = False):
        pooled, captured = Comparison(), []
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
            if cache.stats.lowrank_captured:
                captured.append(sum(cache.stats.lowrank_captured)
                                / len(cache.stats.lowrank_captured))
        return pooled, (sum(captured) / len(captured) if captured else float("nan"))

    # V is held at the base width in every row. Without this the flat control
    # widens both sides while every targeted row touches K alone, and the
    # comparison silently becomes "one bit on K and V" against "a correction on
    # K" -- which is how the first run produced rows whose logit error barely
    # moved: they could not reach the error V was contributing.
    v_bits = args.bits

    def make(k_bits, rank=0, factor_bits=16, wide=(0, 0)):
        return lambda: BlockKVCache(
            KVCacheConfig(enabled=True, decoded_bits=k_bits, masked_bits=k_bits,
                          key_bits=k_bits, value_bits=v_bits,
                          group_size=args.group_size, key_axis=args.key_axis,
                          value_axis="channel", lowrank_rank=rank,
                          lowrank_kinds=("key",),
                          lowrank_factor_bits=factor_bits,
                          wide_channels=wide[0], wide_extra_bits=wide[1],
                          wide_kinds=("key",)),
            n_layers,
        )

    k_over = overhead_bits(args.key_axis, args.group_size, prefix_len, head_dim)
    v_over = overhead_bits("channel", min(args.group_size, head_dim),
                           prefix_len, head_dim)

    # (label, K bits, rank, factor bits, (channels, extra bits))
    rows = [(f"K {args.bits} flat", args.bits, 0, 16, (0, 0)),
            (f"K {args.bits + 1} flat", args.bits + 1, 0, 16, (0, 0))]
    for factor_bits in args.factor_bits:
        for rank in args.ranks:
            rows.append((f"+rank {rank} @{factor_bits}b", args.bits, rank,
                         factor_bits, (0, 0)))
    for n_ch, extra in mixed:
        rows.append((f"+{n_ch}ch +{extra}b", args.bits, 0, 16, (n_ch, extra)))

    print("\n" + "=" * 78)
    print("The correction is on K only: it is the side the softmax "
          "exponentiates, and splitting the budget halves the rank.")

    for mask_ratio in args.mask_ratios:
        batch = canvases(mask_ratio)
        print(f"\n--- prefix mask ratio {mask_ratio:.2f} " + "-" * 37)
        print(f"{'K storage':>16} {'eff.bits':>9} {'compr':>6} {'rel. err':>10} "
              f"{'vs base':>12} {'argmax':>8} {'argmax@k':>8} {'':>5} "
              f"{'slots@k':>8} {'captured':>9}")

        base_row = {}

        def show(name, c, bits, compr, captured=float("nan"), note=""):
            se = c.agree_k_se
            se_txt = "     " if se != se else f"±{100 * se:3.0f}%"
            cap = "        -" if captured != captured else f"{100 * captured:8.1f}%"
            bt = "         " if bits != bits else f"{bits:>9.2f}"
            ct = "      " if compr != compr else f"{compr:>6.2f}"
            # Against the base width, paired canvas by canvas. Each row's own
            # spread is mostly how hard the canvases were, which cancels here.
            delta = "            "
            if base_row and name != base_row["name"]:
                d, dse = c.paired_delta(base_row["c"])
                if d == d:
                    delta = f"{100 * d:+6.1f}±{100 * dse:4.1f}%"
            print(f"{name:>16} {bt} {ct} {c.rel:>10.3e} {delta} {pct(c.agree)} "
                  f"{pct(c.agree_k)} {se_txt} {pct(c.slots_k)} {cap}{note}")

        # The chance floor first. At a high prefix mask ratio `argmax@k` is
        # degenerate -- the measured floor is 68.75% there, against 9.38% on a
        # decoded prefix -- so a row cannot be read without it, and the metric
        # that still discriminates is `slots@k`, whose floor is 12.50%. This
        # matters here specifically: the residual spectrum says the correction
        # has something to recover exactly at full mask, which is the one place
        # the headline metric cannot see it.
        floor, _ = run_row(make(16), batch, mask_ratio, scramble=True)
        show("scrambled", floor, float("nan"), float("nan"),
             note="   <-- chance floor")

        measured = {}
        for label, bits, rank, factor_bits, wide in rows:
            c, captured = run_row(make(bits, rank, factor_bits, wide), batch,
                                  mask_ratio)
            measured[label] = c
            k_bits = (bits + k_over
                      + lowrank_bits(rank, prefix_len, head_dim, factor_bits,
                                     factor_scales=args.factor_scales)
                      + wide[1] * min(wide[0], head_dim) / head_dim)
            mean_bits, compr = pair_cost(k_bits, v_bits + v_over)
            note = ("   <-- at the floor"
                    if c.rel >= floor.rel and c.agree <= floor.agree else "")
            if not base_row:
                base_row.update(name=label, c=c)
            show(label, c, mean_bits, compr, captured, note)

        warn_if_underpowered(measured, args.samples, args.commit_k)

        print("      Compare rows at equal eff.bits, not at equal base width. "
              "A correction that beats its own base while costing more than "
              "the next bit up has not beaten anything.")
        if mask_ratio >= 0.75:
            print("      At this mask ratio read `slots@k`, not `argmax@k`: "
                  "every cached position carries the same embedding row, the "
                  "committed tokens are nearly right whatever the cache does, "
                  "and what breaks is which positions get committed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
