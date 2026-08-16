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
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dllmquant.cache import (  # noqa: E402
    BlockKVCache,
    KVCacheConfig,
    lowrank_bits,
    lowrank_residual,
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
from check_static_scales import TapCache, overhead_bits, pair_cost  # noqa: E402


def rms(x: torch.Tensor) -> float:
    return float(x.float().pow(2).mean().sqrt())


def survey_side(
    tensors: List[torch.Tensor],
    bits: int,
    ranks: List[int],
    axis: str,
    group: int,
    factor_bits: int,
) -> List[Tuple]:
    """Residual energy recovered per rank, priced against one more flat bit.

    Returns one row per rank plus the flat-bit reference, all as fractions of
    the uncorrected residual, so the two are directly comparable and neither
    depends on the scale of the tensor.
    """
    rows = []
    n_tokens, head_dim = tensors[0].shape[-2], tensors[0].shape[-1]

    base, capt, left = [], {r: [] for r in ranks}, {r: [] for r in ranks}
    finer = []
    for t in tensors:
        q = quantize_kv(t, bits, group, axis=axis)
        b = rms(t - q)
        base.append(b)
        for r in ranks:
            corr, c = lowrank_residual(t, q, r, factor_bits)
            capt[r].append(c)
            left[r].append(rms(t - q - corr) / max(b, 1e-12))
        finer.append(rms(t - quantize_kv(t, bits + 1, group, axis=axis)) / max(b, 1e-12))

    def mean(xs):
        return sum(xs) / len(xs)

    flat_over = overhead_bits(axis, group, n_tokens, head_dim)
    for r in ranks:
        cost = lowrank_bits(r, n_tokens, head_dim, factor_bits)
        rows.append((f"rank {r}", mean(capt[r]), mean(left[r]),
                     bits + flat_over + cost))
    rows.append(("+1 bit flat", float("nan"), mean(finer),
                 bits + 1 + flat_over))
    return rows


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
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--key-axis", default="token", choices=["channel", "token"])
    ap.add_argument("--mask-ratios", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--commit-k", type=int, default=4)
    ap.add_argument("--window-mask-ratio", type=float, default=1.0)
    ap.add_argument("--survey-only", action="store_true",
                    help="stop after the residual spectrum. It costs one "
                         "forward per canvas and can close the question.")
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

    # ---- survey ------------------------------------------------------------

    print(f"\n=== residual spectrum, {args.bits} bits, K along "
          f"{args.key_axis}s, group {args.group_size} " + "=" * 12)
    print("captured = share of the residual's energy the correction recovers")
    print("left     = what remains, as a fraction of the uncorrected residual")
    print("bits     = total width per entry, correction and scales included")
    print("The row to beat is the last one in each block: the same question "
          "asked of one more flat bit, which needs no rank and no factors.")

    for mask_ratio in args.mask_ratios:
        collected: Dict[int, List[torch.Tensor]] = {}
        for i in range(args.samples):
            for layer, (k, _) in raw_kv(canvas_for(i, mask_ratio), mask_ratio).items():
                if layer in layers:
                    collected.setdefault(layer, []).append(k)

        print(f"\n--- prefix mask ratio {mask_ratio:.2f} " + "-" * 37)
        for factor_bits in args.factor_bits:
            print(f"\n  factors at {factor_bits} bits")
            print(f"    {'layer':>5} {'correction':>12} {'captured':>9} "
                  f"{'left':>7} {'bits':>7}")
            for layer in sorted(collected):
                rows = survey_side(collected[layer], args.bits, args.ranks,
                                   args.key_axis, args.group_size, factor_bits)
                for name, captured, left, bits in rows:
                    cap = "      -" if captured != captured else f"{100 * captured:6.1f}%"
                    note = ""
                    if name == "+1 bit flat":
                        note = "   <-- the control"
                    print(f"    {layer:>5} {name:>12} {cap:>9} "
                          f"{left:>7.3f} {bits:>7.2f}{note}")
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

    def run_row(make_cache, batch, mask_ratio):
        pooled, captured = Comparison(), []
        for x, reference, committable in batch:
            cache = make_cache()
            cache.mask_ratio = mask_ratio
            for state in states:
                state.cache = cache
            refresh_prefix(model, states, x, prefix_len, args.block_length)
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

    def make(bits, rank=0, factor_bits=16):
        return lambda: BlockKVCache(
            KVCacheConfig(enabled=True, decoded_bits=bits, masked_bits=bits,
                          group_size=args.group_size, key_axis=args.key_axis,
                          value_axis="channel", lowrank_rank=rank,
                          lowrank_kinds=("key",),
                          lowrank_factor_bits=factor_bits),
            n_layers,
        )

    k_over = overhead_bits(args.key_axis, args.group_size, prefix_len, head_dim)
    v_over = overhead_bits("channel", min(args.group_size, head_dim),
                           prefix_len, head_dim)

    rows = [("flat", args.bits, 0, 16), ("flat", args.bits + 1, 0, 16)]
    for factor_bits in args.factor_bits:
        for rank in args.ranks:
            rows.append((f"+rank {rank} @{factor_bits}b", args.bits, rank,
                         factor_bits))

    print("\n" + "=" * 78)
    print("The correction is on K only: it is the side the softmax "
          "exponentiates, and splitting the budget halves the rank.")

    for mask_ratio in args.mask_ratios:
        batch = canvases(mask_ratio)
        print(f"\n--- prefix mask ratio {mask_ratio:.2f} " + "-" * 37)
        print(f"{'K storage':>16} {'eff.bits':>9} {'compr':>6} {'rel. err':>10} "
              f"{'argmax':>8} {'argmax@k':>8} {'':>5} {'captured':>9}")

        results = {}
        for label, bits, rank, factor_bits in rows:
            c, captured = run_row(make(bits, rank, factor_bits), batch, mask_ratio)
            k_bits = bits + k_over + lowrank_bits(rank, prefix_len, head_dim,
                                                  factor_bits)
            mean_bits, compr = pair_cost(k_bits, bits + v_over)
            se = c.agree_k_se
            se_txt = "     " if se != se else f"±{100 * se:3.0f}%"
            cap = "        -" if captured != captured else f"{100 * captured:8.1f}%"
            name = label if rank else f"{label} {bits} bits"
            print(f"{name:>16} {mean_bits:>9.2f} {compr:>6.2f} {c.rel:>10.3e} "
                  f"{pct(c.agree)} {pct(c.agree_k)} {se_txt} {cap}")
            results[(name, round(mean_bits, 2))] = c

        print("      Compare rows at equal eff.bits, not at equal base width. "
              "A correction that beats its own base while costing more than "
              "the next bit up has not beaten anything.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
