"""Is K simply harder to quantize on the MoE, before anything downstream?

Section 2.5a leaves a gap unexplained. The grouping axis costs forty-one
points on LLaDA2.0-mini and about nothing on LLaDA-1.5, and the rounding
floor is five to thirteen times higher on the MoE -- while the structure that
is supposed to explain the axis, K's outliers sitting in fixed channels, is
nearly identical on the two (top-8 channel agreement 0.64-0.70 against
0.66-0.81). Two repairs were tested and both fell: routing is out because K
and V damage equally at 0.09 sigma, and the GQA fan-out is out because
correlated error turned out to be the *less* damaging of the two.

An earlier candidate of mine does not survive arithmetic. I proposed that the
MoE is short of scales -- four K heads against thirty-two. It has eight times
fewer scales and eight times fewer stored values, so scales per value are
equal and there is no scarcity to find.

What is left is a claim about the tensor rather than the architecture. A GQA
key head serves four query subspaces at once, so it may simply carry more and
quantize worse. That is measurable without a sampler, without logits, and
without anything that could confound it: take K after RoPE from the same
layers on both models, quantize it both ways at the same width, and compare
the relative error and the ratio between axes.

    2.0 already has these numbers from 2.1: 8.03e-02 along tokens against
    1.68e-01 along channels, a ratio of 2.14. If LLaDA-1.5's ratio is the
    same, the tensor behaves identically and the whole difference lives
    downstream -- which is a sharper finding than the capacity story ever
    was. If its ratio is much smaller, K on the MoE really is the harder
    tensor and the gap has an explanation at last.

    python scripts/check_key_error.py --model GSAI-ML/LLaDA-1.5 --model-type llada
    bash scripts/llada2.sh scripts/check_key_error.py \\
        --model inclusionAI/LLaDA2.0-mini --model-type llada2_moe
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, List

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dllmquant.cache import quantize_kv  # noqa: E402
from dllmquant.config import DLLMQuantConfig, QuantConfig  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402

from check_block_cache import text_ids  # noqa: E402


def rel_err(q: torch.Tensor, x: torch.Tensor) -> float:
    """Frobenius relative error, the same quantity 2.1 reports."""
    return float((q - x).norm() / x.norm().clamp_min(1e-12))


def crest(x: torch.Tensor) -> float:
    """Peak over RMS, per head, averaged. How outlier-ridden the tensor is."""
    flat = x.reshape(x.shape[1], -1).float()
    return float((flat.abs().amax(-1) / flat.pow(2).mean(-1).sqrt()).mean())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="llada2_moe",
                    choices=["llada2_moe", "llada"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 3])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="which blocks to probe; default is an even spread of "
                         "six, because the two families differ in depth (20 "
                         "against 32) and a fixed index would compare "
                         "different fractions of the stack")
    ap.add_argument("--mask-ratio", type=float, default=0.5,
                    help="share of the canvas held masked. K's statistics move "
                         "along the trajectory (2.4), so the two models have "
                         "to be read at the same point on it")
    args = ap.parse_args()

    cfg = DLLMQuantConfig(model_path=args.model, model_type=args.model_type,
                          dtype=args.dtype, device=args.device,
                          weight=QuantConfig(n_bits=16),
                          activation=QuantConfig(n_bits=16))
    adapter = build_adapter(cfg)
    adapter.load()
    print(adapter.describe())

    blocks = adapter.blocks
    depth = len(blocks)
    idx = args.layers
    if idx is None:
        idx = [round(i * (depth - 1) / 5) for i in range(6)]
    idx = [i for i in idx if 0 <= i < depth]

    print(f"\n{args.samples} canvases of {args.seq_len} tokens, mask ratio "
          f"{args.mask_ratio:.2f}, layers {idx} of {depth}")
    print("K taken after RoPE -- the tensor the cache stores")

    # rel_err[(bits, axis)] -> list over (layer, canvas)
    acc: Dict[tuple, List[float]] = {}
    crests: List[float] = []

    for s in range(args.samples):
        ids = text_ids(adapter, args.seq_len, seed=s).unsqueeze(0)
        keep = int(args.seq_len * (1.0 - args.mask_ratio))
        x = ids.clone()
        x[:, keep:] = adapter.mask_id
        x = x.to(adapter.model.device)

        for li in idx:
            probe = adapter.make_probe(blocks[li])
            # Inside the `with`, not after it: AttentionProbe.__exit__ clears
            # `parts` so a stale capture cannot be read as a fresh one, and
            # reading after the block gets None every time.
            with probe, torch.no_grad():
                adapter.model(x, **adapter.forward_kwargs(x))
                if probe.parts is None:
                    raise SystemExit(
                        f"the hook on block {li} never fired -- the forward "
                        "did not reach it, so there is nothing to measure")
                k = probe.parts.key_states
                if k is None:
                    raise SystemExit(
                        "the probe captured no key_states -- this adapter's "
                        "hook predates the field; nothing below would be "
                        "measuring K")
                k = k.detach().float()
            crests.append(crest(k))
            for bits in args.bits:
                for axis in ("token", "channel"):
                    q = quantize_kv(k, bits, args.group_size, axis=axis)
                    acc.setdefault((bits, axis), []).append(rel_err(q, k))

    def mean(key) -> float:
        v = acc[key]
        return sum(v) / len(v)

    print()
    print("=== relative error of K, one scale per group of "
          f"{args.group_size} ===")
    print(f"{'bits':>5} {'along tokens':>14} {'along channels':>15} "
          f"{'channel/token':>14}")
    print("-" * 52)
    for bits in args.bits:
        t, c = mean((bits, "token")), mean((bits, "channel"))
        print(f"{bits:>5} {t:>14.3e} {c:>15.3e} {c / t:>13.2f}x")

    print()
    print(f"  crest factor of K, peak over RMS per head: {sum(crests) / len(crests):.2f}")
    print()
    print("  Read the ratio, not the level: the level moves with width and")
    print("  with where on the trajectory the canvas sits, the ratio is what")
    print("  says whether K's outliers are concentrated enough for the axis")
    print("  to matter. LLaDA2.0-mini reports 2.14 at four bits on a decoded")
    print("  prefix (2.1). A dense model landing near it says the tensors")
    print("  behave alike and the whole difference is downstream of them --")
    print("  which would leave 2.5a's gap unexplained and say so precisely.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
