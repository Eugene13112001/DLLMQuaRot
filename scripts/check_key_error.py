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

    Compare only against another run of *this* script. §2.1's 8.03e-02
    against 1.68e-01 is a ratio of **logit** error, which is a different
    quantity from the tensor error here; reading one against the other once
    made a broken probe look verified. Measured with QK-Norm applied, the
    MoE's K sits near 4.1-4.8x and LLaDA-1.5's near 2.0x, so the tensors do
    not behave alike and the norm is what separates them.

V and a group-size sweep were added afterwards, to answer a claim that had
been standing on one measured cell. The reading that came out of the first
run -- where the two models see the same perturbation they agree, where they
see different ones the MoE amplifies the difference -- rests on four cases,
and only the K axis was measured on both. The group-size case came from 2.3,
which is MoE-only and taken on decisions; the V case was not measured at all,
only inferred from V having no channel concentration. Both are cheap here:
AttentionParts already carries value_states, and the group size is a loop.

    python scripts/check_key_error.py --model GSAI-ML/LLaDA-1.5 --model-type llada
    bash scripts/llada2.sh scripts/check_key_error.py \\
        --model inclusionAI/LLaDA2.0-mini --model-type llada2_moe
"""

from __future__ import annotations

import argparse
import json
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


def logit_rel_err(q: torch.Tensor, err: torch.Tensor, k: torch.Tensor,
                  allowed) -> float:
    """||q . err^T|| / ||q . k^T|| over the positions attention may read.

    The quantity attention actually receives. After the K-norm gain is moved into
    the Q-norm the denominator is bit-identical, so this number is comparable
    before and after the migration -- the error in K space is not, because the
    loud channels now sit in Q and multiply whatever error K carries there.
    """
    rep = q.shape[1] // err.shape[1]
    if rep > 1:
        err = err.repeat_interleave(rep, dim=1)
        k = k.repeat_interleave(rep, dim=1)
    de = q @ err.transpose(-1, -2)
    sc = q @ k.transpose(-1, -2)
    if allowed is not None:
        de = de * allowed
        sc = sc * allowed
    return float(de.norm() / sc.norm().clamp_min(1e-12))


def qk_geometry(q: torch.Tensor, k: torch.Tensor, allowed) -> tuple:
    """Two numbers about how the logits are built, over the allowed positions.

    mean |cos(q_i, k_j)|, and the isotropic-noise factor
        sqrt(sum_ij ||q_i||^2 ||k_j||^2) / sqrt(sum_ij (q_i . k_j)^2).
    If a quantizer's error in key j is isotropic with a size proportional to
    ||k_j|| -- what a rotation followed by a per-token scale produces -- its
    relative logit error is this factor times a constant of the grid. So the
    factor is the prediction for QuaRot, and it moves when the migration moves
    loudness from K into Q while leaving every q . k where it was.
    """
    rep = q.shape[1] // k.shape[1]
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
    dots = q @ k.transpose(-1, -2)
    qn = q.norm(dim=-1, keepdim=True)
    kn = k.norm(dim=-1, keepdim=True)
    norms = qn @ kn.transpose(-1, -2)
    w = allowed if allowed is not None else torch.ones_like(dots)
    w = w.expand_as(dots)
    cos = (dots.abs() / norms.clamp_min(1e-12) * w).sum() / w.sum().clamp_min(1)
    iso = ((norms ** 2 * w).sum() / ((dots ** 2) * w).sum().clamp_min(1e-24)).sqrt()
    return float(cos), float(iso)


def split_logit_err(q: torch.Tensor, err: torch.Tensor, k: torch.Tensor,
                    allowed, loud: torch.Tensor) -> tuple:
    """The logit error carried by the loud channels and by the rest, each over ||q K^T||.

    ``loud`` marks the channels where the K-norm gain is above the head median --
    the ones the migration shrinks in K and grows in Q. If the error left after
    the migration sits there, it is the moved loudness in Q multiplying K's
    noise; if it sits in the rest, it is outliers the gain did not make.
    """
    rep = q.shape[1] // err.shape[1]
    if rep > 1:
        err = err.repeat_interleave(rep, dim=1)
        k = k.repeat_interleave(rep, dim=1)
    den = q @ k.transpose(-1, -2)
    if allowed is not None:
        den = den * allowed
    den = den.norm().clamp_min(1e-12)
    out = []
    for sel in (loud, ~loud):
        de = q[..., sel] @ err[..., sel].transpose(-1, -2)
        if allowed is not None:
            de = de * allowed
        out.append(float(de.norm() / den))
    return tuple(out)


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
                    choices=["llada2_moe", "llada", "fast_dllm_v2"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 3])
    ap.add_argument("--group-size", type=int, nargs="+", default=[128],
                    help="group sizes to sweep. 2.3 found 32 and 128 "
                         "indistinguishable on the right axis, but only on the "
                         "MoE and only at the decision level -- the tensor side "
                         "of that claim has never been taken on either model")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="which blocks to probe; default is an even spread of "
                         "six, because the two families differ in depth (20 "
                         "against 32) and a fixed index would compare "
                         "different fractions of the stack")
    ap.add_argument("--mask-ratio", type=float, default=0.5,
                    help="share of the canvas held masked. K's statistics move "
                         "along the trajectory (2.4), so the two models have "
                         "to be read at the same point on it")
    ap.add_argument("--skip-qk-norm", action="store_true",
                    help="read K and Q before QK-Norm, the tensor the model "
                         "never caches. It is here for one row only: the "
                         "within-model leg of the QK-Norm finding, same "
                         "checkpoint, same canvases, one variable. Refused on a "
                         "model that has no norm to skip")
    ap.add_argument("--dump", default=None,
                    help="write every (canvas, layer) error to this JSON. The "
                         "table prints ratios of means and nothing about their "
                         "spread; scripts/stats_theses.py resamples canvases "
                         "from the dump for the intervals")
    ap.add_argument("--rotate", action="store_true",
                    help="also quantize K and V after R4 -- the head-wise "
                         "random Hadamard QuaRot puts on Q/K after RoPE, same "
                         "seed as --rotate-qk -- on both axes. This is the "
                         "question the per-channel scale and QuaRot answer "
                         "differently: catch fixed-channel outliers with a "
                         "scale per channel, or smear them with a rotation and "
                         "then group per token the way V is grouped. The "
                         "rotation is orthogonal, so the error measured in the "
                         "rotated frame is the error of the attention input")
    ap.add_argument("--migrate-qk", type=float, default=0.0, metavar="ALPHA",
                    help="move the K-norm gain into the Q-norm first (see "
                         "dllmquant/algos/smooth_qk.py). Attention is unchanged; the keys "
                         "this script measures are the ones the cache would store after the "
                         "migration. The control for the QK-Norm claim, on the tensor")
    ap.add_argument("--logit-error", action="store_true",
                    help="also report the relative error of the attention logits, "
                         "||q (K - Q(K))^T|| / ||q K^T|| on the positions the mask "
                         "lets attention read, for every K scheme (cells 'L/...'). "
                         "This is the number to compare across --migrate-qk: the "
                         "migration leaves q K^T bit-identical but moves the loud "
                         "channels into q, so the K-space error stops meaning what "
                         "attention sees")
    ap.add_argument("--migrate-mode", default="shrink", choices=["shrink", "geomean"],
                    help="how --migrate-qk picks its factor (dllmquant/algos/smooth_qk.py)")
    ap.add_argument("--all-layers", action="store_true",
                    help="probe every block instead of six. The weight-law check "
                         "correlates a per-layer statistic of the K-norm gain with "
                         "the per-layer axis cost, and six points cannot carry that")
    ap.add_argument("--split-rope", action="store_true",
                    help="also report the error restricted to the head channels RoPE "
                         "rotates and to the ones it leaves alone. The whole tensor is "
                         "still quantized as the cache would quantize it; only the "
                         "error is read on each part. On a partially rotated head this "
                         "separates what the norm gain does from what rotary does: "
                         "outliers in the unrotated part cannot come from RoPE")
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
    if args.all_layers:
        idx = list(range(depth))
    if idx is None:
        idx = [round(i * (depth - 1) / 5) for i in range(6)]
    idx = [i for i in idx if 0 <= i < depth]

    print(f"\n{args.samples} canvases of {args.seq_len} tokens, mask ratio "
          f"{args.mask_ratio:.2f}, layers {idx} of {depth}")
    print("K taken after RoPE -- the tensor the cache stores"
          + (" -- except QK-Norm is SKIPPED, so not what it stores"
             if args.skip_qk_norm else ""))

    # Which channels the K-norm gain makes loud, read from the ORIGINAL gains so
    # that the split means the same channels with and without the migration.
    loud_masks: Dict[int, torch.Tensor] = {}
    if args.logit_error:
        from dllmquant.algos.smooth_qk import _norms, gain_scales
        rd0 = adapter._probe_rotary_dim() or adapter.head_dim
        for li in idx:
            _, k_norm = _norms(blocks[li])
            if k_norm is not None:
                s0 = gain_scales(k_norm.weight.detach().float().cpu(), rd0, alpha=1.0)
                loud_masks[li] = s0 > 1.0

    if args.migrate_qk:
        from dllmquant.algos.smooth_qk import migrate_qk_gains
        migrate_qk_gains(adapter, args.migrate_qk, mode=args.migrate_mode)

    rot = None
    if args.rotate:
        from dllmquant.algos.quarot import random_hadamard_matrix
        rot = random_hadamard_matrix(adapter.head_dim, seed=0)
        print("R4 variants on: K and V also quantized after a head-wise "
              "random Hadamard (seed 0, the matrix --rotate-qk installs)")
    axes = ("token", "channel") + (("token+rot", "channel+rot") if rot is not None else ())

    # rel_err[(bits, axis)] -> list over (layer, canvas)
    acc: Dict[tuple, List[float]] = {}
    crests: List[float] = []
    vcrests: List[float] = []
    qk_cos: List[float] = []
    qk_iso: List[float] = []
    # Weights of the Q/K norms per probed layer, and what the probe actually
    # did with rotary -- recorded so the weight-law analysis reads the gains
    # from the same modules that produced the tensor, and a probe that silently
    # skipped RoPE cannot pass for one that applied it.
    gammas: Dict[str, Dict[int, List[float]]] = {"k": {}, "q": {}}
    rotary_dims: Dict[int, int] = {}
    rope_applied: Dict[int, bool] = {}

    for s in range(args.samples):
        ids = text_ids(adapter, args.seq_len, seed=s).unsqueeze(0)
        keep = int(args.seq_len * (1.0 - args.mask_ratio))
        x = ids.clone()
        x[:, keep:] = adapter.mask_id
        x = x.to(adapter.model.device)

        for li in idx:
            probe = adapter.make_probe(blocks[li])
            if s == 0:
                for name, mod in (("k", getattr(probe, "k_norm", None)),
                                  ("q", getattr(probe, "q_norm", None))):
                    w = getattr(mod, "weight", None)
                    if w is not None:
                        gammas[name][li] = [float(x) for x in w.detach().float().cpu()]
                rotary_dims[li] = int(getattr(probe, "rotary_dim", adapter.head_dim))
            if args.skip_qk_norm:
                if getattr(probe, "k_norm", None) is None:
                    raise SystemExit(
                        "--skip-qk-norm on a model with no QK-Norm: the row "
                        "would be the normal one under a misleading label")
                probe.q_norm = probe.k_norm = None
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
                v = probe.parts.value_states.detach().float()
                qs = None
                if args.logit_error:
                    qs = probe.parts.query_states
                    if qs is None:
                        raise SystemExit("the probe captured no query_states; "
                                         "--logit-error has nothing to multiply by")
                    qs = qs.detach().float()
            rope_applied[li] = bool(getattr(probe, "rope_applied", False))
            crests.append(crest(k))
            vcrests.append(crest(v))
            rotated = {}
            if rot is not None:
                h = rot.to(k.device)
                rotated = {"K": k @ h, "V": v @ h}
            allowed = None
            if qs is not None:
                mask_fn = getattr(probe, "attn_mask_fn", None)
                if mask_fn is not None:
                    m = mask_fn(qs.shape[-2], qs.device, torch.float32)
                    if m is not None:
                        allowed = (m > -1e4).float()
            loud = None
            if qs is not None:
                c, iso = qk_geometry(qs, k, allowed)
                qk_cos.append(c)
                qk_iso.append(iso)
                if li in loud_masks:
                    loud = loud_masks[li].to(k.device)
            parts = {}
            if args.split_rope:
                rd = rotary_dims[li]
                d = k.shape[-1]
                parts["rot"] = list(range(min(rd, d)))
                if rd < d:
                    parts["pass"] = list(range(rd, d))
            for bits in args.bits:
                for axis in axes:
                    base, _, turned = axis.partition("+")
                    for g in args.group_size:
                        for side, t0 in (("K", k), ("V", v)):
                            t = rotated[side] if turned else t0
                            q = quantize_kv(t, bits, g, axis=base)
                            acc.setdefault((side, bits, axis, g), []).append(
                                rel_err(q, t))
                            if qs is not None and side == "K":
                                model_frame = q @ h.T if turned else q
                                acc.setdefault(("L", bits, axis, g), []).append(
                                    logit_rel_err(qs, model_frame - t0, t0, allowed))
                                if loud is not None and bool(loud.any()) and bool((~loud).any()):
                                    l_loud, l_rest = split_logit_err(
                                        qs, model_frame - t0, t0, allowed, loud)
                                    acc.setdefault(("Lloud", bits, axis, g), []).append(l_loud)
                                    acc.setdefault(("Lrest", bits, axis, g), []).append(l_rest)
                            if parts:
                                # Read the error in the model's own frame, so a
                                # channel subset means the same channels whether
                                # or not the quantizer rotated first.
                                back = q @ h.T if turned else q
                                for part, sel in parts.items():
                                    acc.setdefault((side, bits, f"{axis}@{part}", g), []).append(
                                        rel_err(back[..., sel], t0[..., sel]))

    def mean(key) -> float:
        v = acc[key]
        return sum(v) / len(v)

    print()
    print("=== relative error, tensor level ===")
    print(f"{'side':>5} {'bits':>5} {'group':>6} {'along tokens':>14} "
          f"{'along channels':>15} {'channel/token':>14}")
    print("-" * 64)
    for side in ("K", "V"):
        for bits in args.bits:
            for g in args.group_size:
                t = mean((side, bits, "token", g))
                c = mean((side, bits, "channel", g))
                print(f"{side:>5} {bits:>5} {g:>6} {t:>14.3e} {c:>15.3e} "
                      f"{c / t:>13.2f}x")

    if rot is not None:
        print()
        print("=== R4 against the per-channel scale (\"along tokens\" = one scale "
              "per channel) ===")
        print(f"{'side':>5} {'bits':>5} {'group':>6} {'ch scale':>10} {'tok scale':>10} "
              f"{'tok+R4':>10} {'ch+R4':>10} {'tok+R4 / ch':>12}")
        print("-" * 76)
        for side in ("K", "V"):
            for bits in args.bits:
                for g in args.group_size:
                    ch = mean((side, bits, "token", g))
                    tok = mean((side, bits, "channel", g))
                    tokr = mean((side, bits, "channel+rot", g))
                    chr_ = mean((side, bits, "token+rot", g))
                    print(f"{side:>5} {bits:>5} {g:>6} {ch:>10.3e} {tok:>10.3e} "
                          f"{tokr:>10.3e} {chr_:>10.3e} {tokr / ch:>11.2f}x")
        print("  tok+R4 / ch above one: the per-channel scale beats QuaRot's "
              "rotate-then-group-per-token on this tensor.")

    if args.logit_error:
        print()
        print("=== relative error of the attention logits, q (K - Q(K))^T / q K^T ===")
        cols = [a for a in axes]
        print(f"{'bits':>5} {'group':>6} " + " ".join(f"{a:>13}" for a in cols)
              + f" {'tok/ch':>8}" + (f" {'quarot/ch':>10}" if rot is not None else ""))
        print("-" * (16 + 14 * len(cols) + 20))
        for bits in args.bits:
            for g in args.group_size:
                vals = {a: mean(("L", bits, a, g)) for a in cols}
                line = f"{bits:>5} {g:>6} " + " ".join(f"{vals[a]:>13.3e}" for a in cols)
                line += f" {vals['channel'] / vals['token']:>7.2f}x"
                if rot is not None:
                    line += f" {vals['channel+rot'] / vals['token']:>9.2f}x"
                print(line)
        print("  'token' = one scale per channel, 'channel' = one per token (the "
              "axis names are the grouping direction). Compare these rows across "
              "--migrate-qk runs: the denominator is the same model.")

    print()
    print(f"  crest factor, peak over RMS per head: "
          f"K {sum(crests) / len(crests):.2f}, V {sum(vcrests) / len(vcrests):.2f}")
    print()
    print("  Two questions this answers that the decision tables cannot. Does")
    print("  the axis change the *tensor* error for V the way it does for K --")
    print("  2.6 says V's axis is indifferent on decisions, and if the tensor")
    print("  error is indifferent too there is simply nothing to amplify. And")
    print("  does the group size change it at all: 2.3 found 32 and 128 alike")
    print("  on decisions, on the MoE, and the tensor side was never taken.")
    print()
    print("  Read the ratio, not the level: the level moves with width and")
    print("  with where on the trajectory the canvas sits, the ratio is what")
    print("  says whether K's outliers are concentrated enough for the axis")
    print("  to matter. Compare it only with another run of this script:")
    print("  2.1's 2.14 is a ratio of *logit* error, a different quantity,")
    print("  and reading one against the other once made a probe that")
    print("  skipped QK-Norm look verified. Measured with the norm applied,")
    print("  LLaDA2.0-mini sits at 4.09x (mask 0.00) to 4.78x (0.50) and")
    print("  LLaDA-1.5 at 1.99x -- the norm is what separates the tensors.")

    if args.dump:
        # Cells are appended canvas by canvas, layer by layer, so the flat
        # lists reshape into [canvas][layer] -- the unit a bootstrap resamples
        # is the canvas, and layers stay inside it as a fixed design.
        n_l = len(idx)

        def grid(v: List[float]) -> List[List[float]]:
            return [v[s * n_l:(s + 1) * n_l] for s in range(args.samples)]

        payload = {
            "config": {
                "model": args.model, "model_type": args.model_type,
                "seq_len": args.seq_len, "samples": args.samples,
                "mask_ratio": args.mask_ratio, "layers": idx,
                "bits": args.bits, "group_size": args.group_size,
                "skip_qk_norm": args.skip_qk_norm, "rotate": args.rotate,
                "split_rope": args.split_rope, "migrate_qk": args.migrate_qk,
                "logit_error": args.logit_error, "migrate_mode": args.migrate_mode,
            },
            "shape": "canvas x layer",
            "errors": {f"{side}/{bits}/{axis}/{g}": grid(v)
                       for (side, bits, axis, g), v in acc.items()},
            "crest": {"K": grid(crests), "V": grid(vcrests)},
            "qk_cos": grid(qk_cos) if qk_cos else None,
            "qk_iso": grid(qk_iso) if qk_iso else None,
            # Per probed layer, in the order of config.layers.
            "gamma": {name: [per.get(li) for li in idx] for name, per in gammas.items()},
            "rotary_dim": [rotary_dims.get(li) for li in idx],
            "rope_applied": [rope_applied.get(li) for li in idx],
        }
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        print(f"\nwrote {args.samples} x {n_l} cells per quantity to {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
