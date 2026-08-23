"""Three fits on the margins ``check_block_reuse --dump-margins`` writes.

The teacher-forced canvas makes a position the same position in every
configuration, so the decision margin can be joined across cells and the
composition of the two cache errors read directly rather than inferred from
flip counts. Flip counts are a thresholded, lossy view of this: a position
counts only once its margin crosses zero, which throws away both the size of
the shift and its sign.

What the fits are for.

**Shrinkage.** 2608.06564 established that quantization does not add noise of a
fixed size -- it multiplies the margin by a factor that falls with the bit
width, median 0.86 at four bits and 0.33 at three, and an additive model lost
in all 378 of their cells. Their evidence is autoregressive and first-token
only; whether the framework transfers to a diffusion model is untested, and
this is the cheapest possible test of it. The same regression on the 16-bit
stale cell gives the coefficient for *staleness*, which is a number nobody has
because nobody else has both errors in one place.

**Additivity.** To first order the margin shift is a linear functional of the
perturbation, so the shifts must add even where the lengths compose in
quadrature. A slope of one confirms the linear regime; a slope below one is
systematic cancellation, a different mechanism with a different prediction.

**Independence of sign.** If the two shifts are uncorrelated their sum behaves
like the quadrature of their magnitudes, which is exactly why the flip count
grows sub-additively while the shifts themselves add. This is the
decision-level counterpart of the cosine ``measure_drift --pin-routes`` reports
on tensors, and the two should agree.

And a fourth question the same dump answers: a confidence-thresholded sampler
commits every position whose top-1 probability clears its threshold, so if
quantization compresses that probability it caps the parallelism the sampler
can reach -- hardest on the first block, where the margin is smallest.

    python scripts/margins.py out/margins.json
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

REFERENCE = "16/every_n:1"
STALE = "16/every_n:2"


def fit(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """Least squares slope, intercept and Pearson r, without numpy."""
    n = len(x)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    if sxx == 0 or syy == 0:
        return float("nan"), float("nan"), float("nan")
    slope = sxy / sxx
    return slope, my - slope * mx, sxy / (sxx * syy) ** 0.5


def sd(v: List[float]) -> float:
    m = sum(v) / len(v)
    return (sum((a - m) ** 2 for a in v) / len(v)) ** 0.5


def pos_of(key: str) -> int:
    return int(key.split(":")[1])


def joined(cells: Dict[str, dict], names: List[str]) -> List[str]:
    """Positions present in every named cell, ordered by prompt then position."""
    keys = set(cells[names[0]])
    for name in names[1:]:
        keys &= set(cells[name])
    return sorted(keys, key=lambda k: (int(k.split(":")[0]), pos_of(k)))


def describe(name: str) -> str:
    bits, policy = name.split("/")
    if policy == "every_n:1":
        return f"rounding alone, {bits} bits"
    if bits == "16":
        return f"staleness alone, {policy}"
    return f"both, {bits} bits, {policy}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", help="margins.json written by check_block_reuse")
    ap.add_argument("--threshold", type=float, default=0.95,
                    help="the confidence a parallel sampler commits above; "
                         "0.95 is what the checkpoint's own generate uses")
    args = ap.parse_args()

    with open(args.dump, encoding="utf-8") as fh:
        payload = json.load(fh)
    cells = payload["cells"]
    cfg = payload["config"]

    print(f"cells: {', '.join(sorted(cells))}")
    print(f"config: {cfg}")

    if REFERENCE not in cells:
        print(f"\nno {REFERENCE} cell: there is nothing to regress against. "
              f"Include 16 in --bits and every_n:1 in --policies.")
        return 1

    names = sorted(cells)
    keys = joined(cells, names)
    print(f"\n{len(keys)} positions present in all {len(names)} cells")

    ref_m = [cells[REFERENCE][k][0] for k in keys]

    # ---------------------------------------------------------------- 1
    print("\n=== 1. shrinkage: a cell's margin against the reference margin ===")
    print("    m_cell = c * m_ref + b;  2608.06564 predicts c = 0.86 at four "
          "bits, 0.33 at three")
    print()
    print(f"{'cell':>16} {'c':>8} {'b':>9} {'r':>7}   what it is")
    print("-" * 70)
    for name in names:
        if name == REFERENCE:
            continue
        y = [cells[name][k][0] for k in keys]
        c, b, r = fit(ref_m, y)
        print(f"{name:>16} {c:>8.3f} {b:>9.3f} {r:>7.3f}   {describe(name)}")

    # ---------------------------------------------------------------- 2, 3
    widths = sorted({n.split("/")[0] for n in names} - {"16"},
                    key=lambda b: -int(b))
    policies = [n.split("/", 1)[1] for n in names if n.startswith("16/")]
    policies = [p for p in policies if p != "every_n:1"]

    if policies and widths:
        print("\n=== 2-3. how the two shifts compose, per staleness row ===")
        print()
        print(f"{'bits':>5} {'policy':>11} {'slope':>8} {'r':>7} {'corr':>7} "
              f"{'sd(dr)':>8} {'sd(ds)':>8} {'sd(both)':>9} {'quadr':>8} "
              f"{'sum':>8} {'saturated':>10}")
        print("-" * 98)
        for policy in policies:
            stale = f"16/{policy}"
            if stale not in cells:
                continue
            d_s = [cells[stale][k][0] - cells[REFERENCE][k][0] for k in keys]
            for bits in widths:
                rnd, both = f"{bits}/every_n:1", f"{bits}/{policy}"
                if rnd not in cells or both not in cells:
                    continue
                d_r = [cells[rnd][k][0] - cells[REFERENCE][k][0] for k in keys]
                d_b = [cells[both][k][0] - cells[REFERENCE][k][0] for k in keys]
                slope, _, r = fit([a + b for a, b in zip(d_r, d_s)], d_b)
                _, _, corr = fit(d_r, d_s)
                sr, ss, sb = sd(d_r), sd(d_s), sd(d_b)
                # Saturation: the margin one error already took, the other
                # cannot take again. Under any additive composition this share
                # stays small; under saturation it is the majority.
                sat = sum(abs(y) < max(abs(a), abs(c))
                          for a, c, y in zip(d_r, d_s, d_b)) / len(keys)
                print(f"{bits:>5} {policy:>11} {slope:>8.3f} {r:>7.3f} "
                      f"{corr:>7.3f} {sr:>8.3f} {ss:>8.3f} {sb:>9.3f} "
                      f"{(sr ** 2 + ss ** 2) ** 0.5:>8.3f} {sr + ss:>8.3f} "
                      f"{100 * sat:>9.0f}%")
        print()
        print("  A slope of 1 would mean the shifts add, as a linear")
        print("  functional must in the small-perturbation regime. corr near")
        print("  zero would mean they compose by quadrature. `saturated` is")
        print("  the share of positions where the two errors together move the")
        print("  margin LESS than the larger of them alone -- the direct")
        print("  signature of a margin that cannot be taken twice.")

    # ---------------------------------------------------------------- 4
    block = cfg.get("block_length", 32)
    first = min(pos_of(k) for k in keys)
    n_blocks = max((pos_of(k) - first) // block for k in keys) + 1
    print(f"\n=== 4. share above confidence {args.threshold}, per block ===")
    print("    what a parallel sampler would commit at that threshold")
    print()
    head = "".join(f"{'blk ' + str(i):>9}" for i in range(n_blocks))
    print(f"{'cell':>16} {head}{'all':>9}")
    print("-" * (17 + 9 * (n_blocks + 1)))
    for name in names:
        counts = [[0, 0] for _ in range(n_blocks)]
        for k in keys:
            b = (pos_of(k) - first) // block
            counts[b][1] += 1
            counts[b][0] += int(cells[name][k][1] > args.threshold)
        row = "".join(f"{100 * h / max(n, 1):>8.1f}%" for h, n in counts)
        tot = sum(h for h, _ in counts) / max(sum(n for _, n in counts), 1)
        print(f"{name:>16} {row}{100 * tot:>8.1f}%")
    print()
    print("  The prediction: block 0 falls with the bit width and the later")
    print("  blocks do not, because the margin there is 3.37 logits against")
    print("  6.4-7.3 later. A fall means a quantized cache caps the")
    print("  parallelism a confidence-thresholded sampler reaches -- which is")
    print("  the number dInfer, Fast-dLLM and DMax sell.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
