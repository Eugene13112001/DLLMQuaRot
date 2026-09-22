"""Pick the noise dose that matches a given centered logit error, per model.

The control for thesis 4 only means something if the dose is matched: LLaDA2.0-mini and
Fast-dLLM-v2 carry the same centered error at four bits per token (0.351 and 0.346) and
score 91.5 and 0.0, so the structureless error has to be brought to that same number on
each model before their answers can be compared. Isotropic noise of relative size sigma
produces a logit error proportional to sigma -- the isotropic-noise factor is the constant
-- so a one-parameter fit through the origin over the sweep gives the dose directly, and
the residuals say whether the proportionality held.

    python scripts/noise_dose.py out/kn_*.json --target 0.35
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


def fit(points: List[Dict]) -> float:
    """Least squares through the origin: err = c * sigma."""
    num = sum(p["sigma"] * p["err"] for p in points)
    den = sum(p["sigma"] ** 2 for p in points)
    return num / den if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--target", type=float, default=0.35,
                    help="centered logit error to match (the four-bit per-token cell)")
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
        c = fit(points)
        if c <= 0:
            print("  no slope: the sweep carries no error, check the dumps")
            continue
        worst = max(abs(p["err"] - c * p["sigma"]) / max(p["err"], 1e-9) for p in points)
        print(f"  err = {c:.2f} * sigma, worst residual {100 * worst:.0f}%"
              + ("" if worst < 0.1 else "  <- not proportional, read the table instead"))
        print(f"  dose for centered error {args.target}: sigma = {args.target / c:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
