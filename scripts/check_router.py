"""Can the MoE router be quantized, or is leaving it in FP16 just caution?

This project excludes the router from quantization by name, on an argument
rather than a measurement: an error in an expert's weights makes an answer
slightly less accurate, while an error in the router sends the token to a
*different expert*, and the router costs 0.06% of the model to keep. The
reasoning is sound and the decision is almost certainly right, but "almost
certainly right" is a policy, not a finding.

This measures it. The router's weight is quantized on its own, everything else
left alone, and three things are reported against the untouched model:

* **routing agreement** -- the fraction of each token's expert set that
  survives. This is the number the policy is about, and it is discrete: a
  router whose logits move by a hair still sends the token somewhere else.
* **relative logit error** and **argmax agreement** at the output, so the
  routing change can be related to what it does to the answer.

The contrast worth drawing is with the measured result that four-bit *expert*
weights cost nothing detectable (McNemar p = 0.44 on GSM8K). If the router at
the same width is visibly worse, the asymmetry is quantified rather than
assumed -- and if it is not, this project is spending FP16 on 19 matrices for
no reason.

    bash scripts/llada2.sh scripts/check_router.py \
        --model inclusionAI/LLaDA2.0-mini --model-type llada2_moe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dllmquant.calib.prompts import load_prompts  # noqa: E402
from dllmquant.config import DLLMQuantConfig, QuantConfig, TMASConfig  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402
from dllmquant.quantizers import quantize_weight_rtn  # noqa: E402
from dllmquant.rotate import routing_fingerprint, routing_overlap  # noqa: E402


def quantize_routers(routers: List[torch.nn.Module], bits: int,
                     group_size: int) -> List[torch.Tensor]:
    """Fake-quantize each router's weight in place; return the originals.

    Round-to-nearest, because that is the honest comparison: the router has no
    calibration data of its own to compensate against, and it is not part of
    the sequential solve. Anything better would be a different experiment.
    """
    cfg = QuantConfig(
        n_bits=bits,
        granularity="per_group" if group_size > 0 else "per_channel",
        group_size=group_size,
        mse_search=False,
    )
    originals = []
    for router in routers:
        w = router.weight.data
        originals.append(w.clone())
        if bits < 16:
            router.weight.data = quantize_weight_rtn(w, cfg).to(w.dtype)
    return originals


def restore_routers(routers: List[torch.nn.Module],
                    originals: List[torch.Tensor]) -> None:
    for router, w in zip(routers, originals):
        router.weight.data = w


@torch.no_grad()
def measure(adapter, x: torch.Tensor):
    logits = adapter.model(x, **adapter.forward_kwargs(x)).logits.float()
    return logits, routing_fingerprint(adapter, x)


def compare(reference, actual):
    ref_logits, ref_routes = reference
    logits, routes = actual
    denom = ref_logits.abs().mean().clamp(min=1e-8)
    rel = float((logits - ref_logits).abs().mean() / denom)
    argmax = float((logits.argmax(-1) == ref_logits.argmax(-1)).float().mean())
    kept = routing_overlap(ref_routes, routes) if ref_routes is not None else float("nan")
    return rel, argmax, kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="llada2_moe", choices=["llada2_moe"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bits", type=int, nargs="+", default=[16, 8, 6, 4, 3, 2])
    ap.add_argument("--group-size", type=int, default=-1,
                    help="-1 for one scale per expert row, which is how the "
                         "rest of this project quantizes weights by default")
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--mask-ratios", type=float, nargs="+", default=[1.0, 0.5, 0.0])
    args = ap.parse_args()

    cfg = DLLMQuantConfig(
        model_path=args.model, model_type=args.model_type,
        dtype=args.dtype, device=args.device,
    )
    adapter = build_adapter(cfg)
    adapter.load()
    print(adapter.describe())

    routers = adapter.routers()
    if not routers:
        print("no routers found -- this is not an MoE checkpoint")
        return 1
    params = sum(r.weight.numel() for r in routers)
    total = sum(p.numel() for p in adapter.model.parameters())
    print(f"\n{len(routers)} routers, {params / 1e6:.2f}M parameters "
          f"= {100 * params / total:.3f}% of the model")

    device = next(adapter.model.parameters()).device
    tmas = TMASConfig(gen_length=args.gen_length, block_length=args.block_length,
                      steps=args.steps, n_samples=args.steps, n_prompts=1)
    prompt = adapter.encode_prompts(load_prompts(1), max_len=64)[0]
    snapshots = adapter.trace_generation(prompt, tmas)

    # One state per requested mask ratio, taken from a real trajectory rather
    # than assembled by hand: what a masked position's K/V look like depends on
    # how the trajectory got there.
    chosen = {}
    for target in args.mask_ratios:
        best = min(snapshots, key=lambda s: abs(s.mask_ratio - target))
        chosen[target] = best

    for target, snap in chosen.items():
        x = snap.input_ids.unsqueeze(0).to(device)
        reference = measure(adapter, x)

        print(f"\n--- mask ratio {snap.mask_ratio:.2f} (asked {target:.2f}) "
              + "-" * 26)
        print(f"{'bits':>5} {'routing kept':>13} {'rel. logit error':>18} "
              f"{'argmax kept':>13}")

        for bits in args.bits:
            originals = quantize_routers(routers, bits, args.group_size)
            try:
                rel, argmax, kept = compare(reference, measure(adapter, x))
            finally:
                restore_routers(routers, originals)

            flag = ""
            if bits >= 16 and (kept < 1.0 or rel > 1e-6):
                flag = "   <-- 16 bits must be a no-op; the harness is wrong"
            print(f"{bits:>5} {100 * kept:>12.2f}% {rel:>18.3e} "
                  f"{100 * argmax:>12.2f}%{flag}")

    print("\nThe 16-bit row is the control: nothing is quantized there, so any "
          "movement is the measurement itself.\n"
          "Read the routing column first. Four-bit expert weights cost nothing "
          "detectable on this model (McNemar p = 0.44); if the router at four "
          "bits keeps far fewer routes, the exclusion is earned. If it keeps "
          "almost all of them, 19 matrices are being held in FP16 out of "
          "caution rather than need.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
