"""Pick the noise dose that matches a given error or a given attention movement.

The control for thesis 4 only means something if the dose is matched, and the tensor run
says on what. The two models carry the same centered logit error at four bits per token
(0.351 and 0.346) and score 91.5 and 0.0, but they do not carry the same *movement*:
Fast-dLLM-v2's attention travels twice as far for that error (KL 0.470 against 0.222),
because its attention is the most peaked of the three. That is one of the two factors, and
it is already measured. What is left is whether, at equal movement, the two models still
differ -- at KL 0.8 LLaDA2.0-mini keeps 59.5 while Fast-dLLM-v2 is at 0.5 with half that.
So the dose worth spending a run on is the one matched on KL, and matching on the error
instead would re-ask the question the tensor already answered.

Isotropic noise of relative size sigma gives a logit error proportional to sigma, and a KL
that grows with its square, so both are one-parameter fits through the origin; the
residuals say whether the form held over the sweep.

    python scripts/noise_dose.py out/kn_*.json --target-kl 0.47 0.22
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


def read(path: str) -> Dict:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    cfg = d["config"]
    sigma = cfg.get("key_noise") or 0.0
    if not sigma:
        raise SystemExit(f"{path}: taken without --key-noise, nothing to calibrate")
    bits = cfg["bits"][0]
    group = cfg["group_size"][0]
    out = {"name": os.path.basename(path).replace(".json", ""),
           "model": cfg["model"].split("/")[-1], "sigma": sigma, "bits": bits}
    for label, key in (("err", "Lc"), ("kl", "KL"), ("flip", "Flip"), ("eff", "Eff")):
        k = f"{key}/{bits}/token/{group}"
        rows = d["errors"].get(k)
        out[label] = (sum(v for r in rows for v in r) / sum(len(r) for r in rows)
                      if rows else None)
    return out


def fit(points: List[Dict], key: str = "err", power: int = 1) -> float:
    """Least squares through the origin: ``key`` = c * sigma**power."""
    xs = [p["sigma"] ** power for p in points]
    ys = [p[key] for p in points]
    den = sum(x * x for x in xs)
    return sum(x * y for x, y in zip(xs, ys)) / den if den else 0.0


def residual(points: List[Dict], c: float, key: str, power: int) -> float:
    return max(abs(p[key] - c * p["sigma"] ** power) / max(p[key], 1e-9) for p in points)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--target", type=float, default=0.35,
                    help="centered logit error to match (the four-bit per-token cell)")
    ap.add_argument("--target-kl", type=float, nargs="*", default=(0.47, 0.22),
                    help="attention movements to match, in KL. The defaults are the two "
                         "levels the four-bit per-token cells sit at: the one that kills "
                         "Fast-dLLM-v2 and the one LLaDA2.0-mini carries unharmed")
    args = ap.parse_args()

    runs = [read(p) for p in args.dumps]
    by_model: Dict[str, List[Dict]] = {}
    for r in runs:
        by_model.setdefault(r["model"], []).append(r)

    for model, points in by_model.items():
        points.sort(key=lambda p: p["sigma"])
        print(f"\n{model}  ({len(points)} doses, {points[0]['bits']} bits)")
        print(f"  {'sigma':>7}{'centered err':>14}{'KL':>11}{'flip':>8}{'eff':>8}")
        for p in points:
            kl = f"{p['kl']:>11.5f}" if p["kl"] is not None else f"{'--':>11}"
            fl = f"{p['flip']:>8.3f}" if p["flip"] is not None else f"{'--':>8}"
            ef = f"{p['eff']:>8.1f}" if p["eff"] is not None else f"{'--':>8}"
            print(f"  {p['sigma']:>7.3f}{p['err']:>14.3f}{kl}{fl}{ef}")
        if len(points) < 3:
            print(f"  only {len(points)} dose(s): a fit through the origin cannot be "
                  "checked here, and the residuals below mean nothing")
        c = fit(points)
        if c <= 0:
            print("  no slope: the sweep carries no error, check the dumps")
            continue
        worst = residual(points, c, "err", 1)
        print(f"  err = {c:.2f} * sigma, worst residual {100 * worst:.0f}%"
              + ("" if worst < 0.1 else "  <- not proportional, read the table instead"))
        print(f"  dose for centered error {args.target}: sigma = {args.target / c:.4f}")
        if any(p["kl"] is None for p in points):
            print("  no KL in these dumps: re-take them with a build that records it")
            continue
        a = fit(points, "kl", 2)
        if a <= 0:
            continue
        wkl = residual(points, a, "kl", 2)
        print(f"  KL  = {a:.2f} * sigma^2, worst residual {100 * wkl:.0f}%"
              + ("" if wkl < 0.15 else "  <- read the table instead"))
        top = max(p["sigma"] for p in points)
        for t in args.target_kl:
            d = (t / a) ** 0.5
            far = d / top
            note = ""
            if far > 1.5:
                note = (f"  <- {far:.1f}x beyond the largest dose measured; KL flattens as "
                        "attention is destroyed, so this is a floor on the dose, not the dose")
            print(f"  dose for movement KL {t}: sigma = {d:.4f}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
