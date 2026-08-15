"""How fast do the latents move, and what would a cache have cost?

Two questions, one trajectory, both models.

**Latent drift.** A block's input at step `t` against the same block's input at
step `t+d`. This is the quantity the cache-reuse literature is implicitly
betting on: reuse is only cheap because the state barely moves. Whether it
barely moves is measurable, and whether it moves *more with depth* decides
whether one refresh interval can serve every layer.

**Cache drift, decomposed.** What a cache would have cost, measured without
building one. Cache K/V at step `t`, read them at `t+d`, and the entry is
wrong for two independent reasons:

    staleness    the state moved between t and t+d
    rounding     the entry was stored in four bits

Both are computable from captured tensors alone: staleness is K(t) against
K(t+d), rounding is K(t) against its own quantized copy. No cache
implementation is required, which is what lets this run on LLaDA-1.5 -- a
model that structurally *cannot* have a cache, because its attention is fully
bidirectional and nothing is ever final.

That contrast is the point of running both. LLaDA2.0 is block-causal, so
positions in already-closed blocks cannot change while the current block is
decoded: their staleness must come out at zero. LLaDA-1.5 has no such
positions and should show drift everywhere. If the numbers say otherwise, an
assumption this project rests on is wrong.

    bash scripts/llada2.sh scripts/measure_drift.py \
        --model inclusionAI/LLaDA2.0-mini --model-type llada2_moe

    python scripts/measure_drift.py --model GSAI-ML/LLaDA-1.5 --model-type llada
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dllmquant.cache import quantize_kv  # noqa: E402
from dllmquant.calib.prompts import load_prompts  # noqa: E402
from dllmquant.config import DLLMQuantConfig, TMASConfig  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative RMS difference, against b's own scale."""
    denom = b.float().pow(2).mean().sqrt().clamp(min=1e-8)
    return float((a.float() - b.float()).pow(2).mean().sqrt() / denom)


@torch.no_grad()
def capture(adapter, x: torch.Tensor, layers: List[int], max_positions: int):
    """One forward; per-layer block input, and the K/V that input produces.

    K/V come from the same probe IA-AQ uses, so fused or split projections,
    grouped-query attention and partial rotary are all handled by code that
    the selfcheck has already exercised against both checkpoints.
    """
    blocks = adapter.blocks
    grabbed: Dict[int, torch.Tensor] = {}

    def make(i: int):
        def hook(module, args, kwargs, output):
            h = kwargs.get("hidden_states", args[0] if args else None)
            if isinstance(h, torch.Tensor) and h.dim() == 3:
                grabbed[i] = h.detach()
        return hook

    handles = [
        blocks[i].register_forward_hook(make(i), with_kwargs=True) for i in layers
    ]
    try:
        adapter.model(x, **adapter.forward_kwargs(x))
    finally:
        for h in handles:
            h.remove()

    out = {}
    total = x.shape[-1]
    keep = torch.arange(total)
    if total > max_positions:
        keep = torch.linspace(0, total - 1, max_positions).long()

    for i in layers:
        hidden = grabbed[i]
        probe = adapter.make_probe(blocks[i])
        _, k, v = probe._project(hidden)  # [B, kv_heads, T, head_dim]
        out[i] = {
            "hidden": hidden[:, keep].float().cpu(),
            "k": k[:, :, keep].float().cpu(),
            "v": v[:, :, keep].float().cpu(),
        }
    return out, keep


def block_of(position: int, bounds) -> int:
    for idx, (lo, hi) in enumerate(bounds):
        if lo <= position < hi:
            return idx
    return -1  # before the first decoding block: prompt, always closed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="llada", choices=["llada", "llada2_moe"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--layer-stride", type=int, default=4,
                    help="measure every Nth block; depth is the axis, not "
                         "every individual layer")
    ap.add_argument("--max-positions", type=int, default=64,
                    help="positions kept per captured state, evenly spaced")
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 2, 4],
                    help="step distances to age a cached entry by")
    ap.add_argument("--bits", type=int, default=4,
                    help="storage width for the rounding half of the split")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--key-axis", default="token", choices=["channel", "token"],
                    help="direction a group runs in for K. Measured to matter "
                         "more than any other knob on this cache: along "
                         "channels, four bits cost twelve points of committed "
                         "decisions; along tokens, nothing. Defaults to the "
                         "better one here -- an earlier default of 'channel' "
                         "inflated the rounding column and understated how far "
                         "staleness dominates it.")
    args = ap.parse_args()

    cfg = DLLMQuantConfig(
        model_path=args.model, model_type=args.model_type,
        dtype=args.dtype, device=args.device,
    )
    adapter = build_adapter(cfg)
    adapter.load()
    print(adapter.describe())

    tmas = TMASConfig(
        gen_length=args.gen_length, block_length=args.block_length,
        steps=args.steps, n_samples=args.steps, n_prompts=1,
    )
    prompt = adapter.encode_prompts(load_prompts(1), max_len=64)[0]
    snapshots = adapter.trace_generation(prompt, tmas)
    total, bounds = adapter._sequence_layout(prompt.shape[-1], tmas)

    layers = list(range(0, len(adapter.blocks), args.layer_stride))
    device = next(adapter.model.parameters()).device
    print(f"\n{len(snapshots)} steps, {len(layers)} of {len(adapter.blocks)} layers "
          f"measured, sequence {total} tokens, blocks {len(bounds)}")

    states = []
    for snap in snapshots:
        x = snap.input_ids.unsqueeze(0).to(device)
        captured, keep = capture(adapter, x, layers, args.max_positions)
        states.append({"data": captured, "keep": keep, "snap": snap})
        print(f"  captured step {snap.step + 1}/{len(snapshots)} "
              f"(mask ratio {snap.mask_ratio:.2f})")

    report_latent_drift(states, layers, args.deltas)
    report_cache_drift(states, layers, args, bounds)
    return 0


def report_latent_drift(states, layers, deltas) -> None:
    """Does the block input move, and does it move more the deeper you go?"""
    print("\n=== latent drift: block input at step t vs t+d " + "=" * 22)
    header = "layer " + "".join(f"{'d=' + str(d):>12}" for d in deltas)
    print(header)

    for layer in layers:
        row = f"{layer:>5}"
        for d in deltas:
            vals = [
                rel(states[t + d]["data"][layer]["hidden"],
                    states[t]["data"][layer]["hidden"])
                for t in range(len(states) - d)
            ]
            row += f"{sum(vals) / max(len(vals), 1):>12.4f}" if vals else f"{'-':>12}"
        print(row)


def report_cache_drift(states, layers, args, bounds) -> None:
    """Staleness against rounding, per layer, and split by block status.

    The prefix/current split is the structural claim under test. A position in
    a block that closed before the current one cannot change under block-causal
    attention, so its staleness must be zero -- not small, zero. Under full
    bidirectional attention there is no such guarantee and the same column
    should be non-zero everywhere.
    """
    # The axis belongs in the header, not only in the flags: it halves the
    # rounding column, and a table that does not say which one produced it
    # cannot be compared with another. This is the second time a heading was
    # left behind while the numbers underneath changed.
    print(f"\n=== cache drift: {args.bits}-bit storage, group "
          f"{args.group_size}, K along {args.key_axis}s "
          + "=" * 14)
    print("staleness = the state moved; rounding = what four bits cost")
    print(f"{'layer':>5} {'d':>3} {'stale(prefix)':>14} {'stale(current)':>15} "
          f"{'rounding':>10} {'ratio':>8}")

    for layer in layers:
        for d in args.deltas:
            prefix_vals, current_vals, round_vals = [], [], []
            for t in range(len(states) - d):
                old = states[t]["data"][layer]
                new = states[t + d]["data"][layer]
                keep = states[t]["keep"]
                block_idx = states[t]["snap"].block_idx

                # Which captured positions belong to blocks already closed at t.
                closed = torch.tensor(
                    [block_of(int(p), bounds) < block_idx for p in keep]
                )
                if closed.any():
                    prefix_vals.append(
                        rel(old["k"][:, :, closed], new["k"][:, :, closed])
                    )
                if (~closed).any():
                    current_vals.append(
                        rel(old["k"][:, :, ~closed], new["k"][:, :, ~closed])
                    )
                round_vals.append(
                    rel(quantize_kv(old["k"], args.bits, args.group_size,
                                    axis=args.key_axis), old["k"])
                )

            def mean(v):
                return sum(v) / len(v) if v else float("nan")

            stale_p, stale_c, rnd = mean(prefix_vals), mean(current_vals), mean(round_vals)
            ratio = stale_c / rnd if rnd > 0 else float("nan")
            print(f"{layer:>5} {d:>3} {stale_p:>14.5f} {stale_c:>15.5f} "
                  f"{rnd:>10.5f} {ratio:>8.2f}")

    print("\nRead the two staleness columns against each other. For a "
          "block-causal model the prefix column is the exactness claim in "
          "numbers, and the current-block column is the error a cache would "
          "actually incur. Where the ratio is below 1, four-bit storage costs "
          "more than the staleness it is being blamed for.")


if __name__ == "__main__":
    raise SystemExit(main())
