"""Does quantizing activations break the MoE's arithmetic, or its dispatch?

Per-token activation quantization costs LLaDA2.0-mini **85 points** of GSM8K
against 58.5 on the dense LLaDA-1.5, and the shape of the failure is not
"answers slightly wrong": only **6%** of replies reach an answer at all,
against 73% on the dense model. A model that has lost some precision produces
worse arithmetic; a model that has lost its routing produces text from the
wrong experts, which is what running out of canvas mid-sentence looks like.

Those two readings predict different things, and the router settles it. If
dispatch survives, the collapse is arithmetic and bit width is the lever. If
dispatch is near the floor, the collapse is routing, and the lever is the
grouping of activations rather than their width -- which is exactly what the
52-to-85-point gap between per-token and group-128 already hints at, without
being able to say why.

**Weights are left alone on purpose.** The claim under test is about
activations, and quantizing weights too would need a GPTQ solve of several
hours and would leave two suspects in the room. Activations are quantized on
the fly by ``QuantLinear``, so this runs in minutes.

The floor is measured, not assumed: the same comparison is run against a
random re-draw of the expert sets, which is what "dispatch is destroyed"
would look like numerically.

    bash scripts/llada2.sh scripts/check_dispatch.py \
        --model inclusionAI/LLaDA2.0-mini --model-type llada2_moe \
        --a-bits 4 --a-group-sizes -1 128
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, List

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dllmquant.config import DLLMQuantConfig, QuantConfig, TMASConfig  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402
from dllmquant.modules import wrap_linears  # noqa: E402

from check_block_cache import text_ids  # noqa: E402
from measure_drift import record_routes  # noqa: E402
from trace_window_path import route_overlap  # noqa: E402


def quantize_activations(adapter, a_bits: int, group_size: int) -> None:
    """Wrap every linear so its *input* is quantized, weights untouched.

    ``skip`` keeps the router itself in full precision. Quantizing it would
    answer a different question -- ``check_router.py`` asks that one -- and
    here it would make the result circular: of course dispatch moves if the
    gate's own arithmetic is perturbed. What is being tested is whether
    perturbing everything *else* is enough to move it.
    """
    w_cfg = QuantConfig(n_bits=16)
    a_cfg = (
        QuantConfig(n_bits=a_bits, granularity="per_group",
                    group_size=group_size, dynamic=True)
        if group_size > 0
        else QuantConfig(n_bits=a_bits, granularity="per_token")
    )
    from trace_window_path import ROUTER_NAMES

    def skip(name: str) -> bool:
        return name.split(".")[-1] in ROUTER_NAMES

    for block in adapter.blocks:
        wrap_linears(block, w_cfg, a_cfg, skip=skip)


def shuffled_like(routes: Dict[int, torch.Tensor],
                  n_experts: int,
                  seed: int = 0) -> Dict[int, torch.Tensor]:
    """A random re-draw of every token's expert set: the destroyed-dispatch floor.

    Not zero. With k of n experts picked at random, two independent draws
    still share k/n of their picks by chance, and on this checkpoint that is
    a few percent -- small, but a row at that level carries no information and
    should not be read as "a fifth survived".
    """
    g = torch.Generator().manual_seed(seed)
    out = {}
    for layer, idx in routes.items():
        rows, k = idx.shape
        out[layer] = torch.stack([
            torch.randperm(n_experts, generator=g)[:k] for _ in range(rows)
        ]).to(idx.device)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="llada2_moe", choices=["llada2_moe"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--a-bits", type=int, default=4)
    ap.add_argument("--a-group-sizes", type=int, nargs="+", default=[-1, 128],
                    help="-1 is one scale per token across all channels, which "
                         "is the setting that collapsed; 128 is the grouping "
                         "that did not")
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--mask-ratios", type=float, nargs="+",
                    default=[1.0, 0.5, 0.0],
                    help="how much of the canvas is still masked. Dispatch "
                         "depends on the whole canvas, so a single ratio would "
                         "describe one moment of the trajectory")
    args = ap.parse_args()

    cfg = DLLMQuantConfig(model_path=args.model, model_type=args.model_type,
                          dtype=args.dtype, device=args.device,
                          weight=QuantConfig(n_bits=16),
                          activation=QuantConfig(n_bits=16))
    adapter = build_adapter(cfg)
    adapter.load()
    print(adapter.describe())

    device = next(adapter.model.parameters()).device
    total = args.prompt_tokens + args.gen_length
    prompt = text_ids(adapter, args.prompt_tokens, seed=0).unsqueeze(0).to(device)
    filler = text_ids(adapter, args.gen_length, seed=1).unsqueeze(0).to(device)

    canvases: Dict[float, torch.Tensor] = {}
    for ratio in args.mask_ratios:
        x = torch.full((1, total), adapter.mask_id, dtype=torch.long,
                       device=device)
        x[:, :args.prompt_tokens] = prompt
        keep = int(round(args.gen_length * (1.0 - ratio)))
        if keep:
            x[:, args.prompt_tokens:args.prompt_tokens + keep] = filler[:, :keep]
        canvases[ratio] = x

    print(f"\nactivations at {args.a_bits} bits, weights and router untouched")
    print(f"canvas {total} tokens, mask ratios "
          + ", ".join(f"{r:.2f}" for r in args.mask_ratios))

    reference = {r: record_routes(adapter, x) for r, x in canvases.items()}
    n_layers = len(next(iter(reference.values())))
    print(f"reference routes captured on {n_layers} routed layers")

    n_experts = None
    for block in adapter.blocks:
        for module in block.modules():
            if hasattr(module, "num_experts"):
                n_experts = int(module.num_experts)
                break
        if n_experts:
            break

    rows: List[tuple] = []
    for gs in args.a_group_sizes:
        # Reloaded per setting: wrap_linears replaces modules in place, and
        # unwrapping is not the inverse of wrapping once a second quantizer
        # has been layered on top of the first.
        adapter = build_adapter(cfg)
        adapter.load()
        quantize_activations(adapter, args.a_bits, gs)
        label = "per token" if gs <= 0 else f"group {gs}"
        for ratio, x in canvases.items():
            got = record_routes(adapter, x)
            overlaps = [route_overlap(reference[ratio][i], got[i])
                        for i in sorted(got) if i in reference[ratio]]
            overlaps = [o for o in overlaps if o == o]
            rows.append((label, ratio, sum(overlaps) / max(len(overlaps), 1),
                         min(overlaps) if overlaps else float("nan")))

    floor = float("nan")
    if n_experts:
        r0 = args.mask_ratios[0]
        rnd = shuffled_like(reference[r0], n_experts)
        vals = [route_overlap(reference[r0][i], rnd[i]) for i in sorted(rnd)]
        floor = sum(vals) / max(len(vals), 1)

    print()
    print("=== expert-set agreement against the unquantized run ===")
    print(f"{'activations':>14} {'mask':>6} {'mean':>8} {'worst layer':>12}")
    print("-" * 44)
    for label, ratio, mean_o, worst in rows:
        print(f"{label:>14} {ratio:>6.2f} {100 * mean_o:>7.1f}% "
              f"{100 * worst:>11.1f}%")
    if floor == floor:
        print(f"{'random re-draw':>14} {'--':>6} {100 * floor:>7.1f}% "
              f"{'--':>12}   <-- destroyed-dispatch floor")

    print()
    print("  Agreement near the floor means per-token quantization is not")
    print("  making the model worse at arithmetic -- it is sending tokens to")
    print("  different experts, and the 6% of replies that reach an answer is")
    print("  what that looks like downstream. Agreement near 100% would send")
    print("  the explanation back to precision, where bit width is the lever.")
    print()
    print("  The router is left in full precision here on purpose: perturbing")
    print("  the gate itself would move dispatch by construction. What is")
    print("  being asked is whether perturbing everything else suffices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
