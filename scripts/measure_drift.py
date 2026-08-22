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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_window_path import ROUTER_NAMES, force_routes  # noqa: E402


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


def record_routes(adapter, x: torch.Tensor):
    """Every router's expert choice on one forward, keyed by layer.

    Only the integer output is kept: that is the choice, and it is the one
    thing ``force_routes`` overrides. Everything downstream of it -- gathering
    the scores, normalising them, the routed scaling -- keeps running in the
    checkpoint's own code, so pinning changes what is selected and nothing
    about how the selection is used.
    """
    routes: Dict[int, torch.Tensor] = {}
    handles = []

    def hook(i):
        def fn(module, inputs, output):
            parts = output if isinstance(output, (tuple, list)) else (output,)
            for t in parts:
                if (isinstance(t, torch.Tensor) and t.numel()
                        and not t.is_floating_point()):
                    routes[i] = t.detach()
                    return
        return fn

    for i, block in enumerate(adapter.blocks):
        for name, module in block.named_modules():
            if (name.split(".")[-1] in ROUTER_NAMES
                    and hasattr(module, "num_experts")):
                handles.append(module.register_forward_hook(hook(i)))
    try:
        adapter.model(x, **adapter.forward_kwargs(x))
    finally:
        for h in handles:
            h.remove()
    return routes


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
    ap.add_argument("--pin-routes", action="store_true",
                    help="hold every router to the choice it made on the first "
                         "captured step. The prefix column is the exactness "
                         "claim of this project in numbers, and a MoE puts a "
                         "second thing between the mask and that number: an "
                         "expert sees the batch its router gathered, so when "
                         "the current block's tokens change, the reduction "
                         "order changes with them and a closed token's output "
                         "moves in arithmetic even though the mask makes it "
                         "independent in exact arithmetic. Pinning removes "
                         "that channel. If the prefix column then goes to "
                         "zero, the residue was dispatch and the structural "
                         "claim stands; if it does not, something reaches "
                         "closed blocks that should not.")
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

    restore = None
    if args.pin_routes:
        first = snapshots[0].input_ids.unsqueeze(0).to(device)
        routes = record_routes(adapter, first)
        restore = force_routes(adapter.blocks, routes, 0, total)
        print(f"  routes pinned from step 1 on {len(routes)} layers")

    states = []
    try:
        for snap in snapshots:
            x = snap.input_ids.unsqueeze(0).to(device)
            captured, keep = capture(adapter, x, layers, args.max_positions)
            states.append({"data": captured, "keep": keep, "snap": snap})
            print(f"  captured step {snap.step + 1}/{len(snapshots)} "
                  f"(mask ratio {snap.mask_ratio:.2f})")
    finally:
        if restore is not None:
            restore()

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

    geometry_table(args, states, layers, bounds)


def geometry_table(args, states, layers, bounds) -> None:
    """Do the two displacements point the same way, or sideways?

    What a reader of the cache gets back decomposes exactly:

        Q(K_old) - K_new  =  [Q(K_old) - K_old]  +  [K_old - K_new]
                               rounding              staleness

    Two vectors in one space, added. The table above gives their lengths; it
    cannot give the angle, and the angle decides how they compose. In a space
    of this dimension two unrelated vectors are nearly orthogonal, so lengths
    add in quadrature rather than linearly -- and with staleness four to ten
    times the longer, adding rounding on top lengthens the total by a few
    percent rather than by its own length. That is a mechanism for the
    absorption measured on decisions, and it is checkable here rather than
    argued.

    Three columns settle it. ``cos`` against zero: a systematic positive value
    means the errors reinforce, negative means they cancel, zero means they
    compose by quadrature. ``|r+s|`` measured against the quadrature
    prediction: if quadrature holds these agree and the ratio sits at 1. The
    null for the cosine concentrates at zero with width about 1/sqrt(D), and D
    here is thousands of elements, so anything past a few hundredths is far
    outside chance.
    """
    print(f"\n=== geometry of the two displacements: {args.bits}-bit storage, "
          f"group {args.group_size}, K along {args.key_axis}s " + "=" * 6)
    print("r = rounding of the stale entry, s = staleness; current block only")
    print(f"{'layer':>5} {'d':>3} {'|r|':>9} {'|s|':>9} {'cos':>8} "
          f"{'|r+s|':>9} {'quadrature':>11} {'ratio':>7}")

    for layer in layers:
        for d in args.deltas:
            rn, sn, cs, tot, quad = [], [], [], [], []
            for t in range(len(states) - d):
                old = states[t]["data"][layer]
                new = states[t + d]["data"][layer]
                keep = states[t]["keep"]
                block_idx = states[t]["snap"].block_idx
                closed = torch.tensor(
                    [block_of(int(p), bounds) < block_idx for p in keep]
                )
                if not (~closed).any():
                    continue
                # The current block only: the prefix carries no staleness under
                # a block-causal mask, so an angle there would be an angle
                # against the zero vector.
                k_old = old["k"][:, :, ~closed].float()
                k_new = new["k"][:, :, ~closed].float()
                k_q = quantize_kv(old["k"], args.bits, args.group_size,
                                  axis=args.key_axis)[:, :, ~closed].float()

                r = (k_q - k_old).flatten()
                sv = (k_old - k_new).flatten()
                nr, ns = r.norm(), sv.norm()
                if nr == 0 or ns == 0:
                    continue
                rn.append(float(nr))
                sn.append(float(ns))
                cs.append(float(torch.dot(r, sv) / (nr * ns)))
                tot.append(float((r + sv).norm()))
                quad.append(float(torch.sqrt(nr ** 2 + ns ** 2)))

            if not cs:
                continue

            def mean(v):
                return sum(v) / len(v)

            m_tot, m_quad = mean(tot), mean(quad)
            print(f"{layer:>5} {d:>3} {mean(rn):>9.4f} {mean(sn):>9.4f} "
                  f"{mean(cs):>8.4f} {m_tot:>9.4f} {m_quad:>11.4f} "
                  f"{m_tot / m_quad:>7.4f}")

    print("\nA cos near zero with the last ratio near 1.0 is quadrature: the "
          "smaller\ndisplacement adds almost nothing to the total, which is why "
          "buying it down\nwith bits buys so little once staleness is present. A "
          "systematically negative\ncos would be cancellation instead -- a "
          "different mechanism with a different\nprediction, and one the "
          "decision-level tables cannot separate on their own.")


if __name__ == "__main__":
    raise SystemExit(main())
