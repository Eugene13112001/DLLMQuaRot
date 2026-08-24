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
    ap.add_argument("--allow-legacy-margins", action="store_true",
                    help="read a dump whose margins are the cell's own top-2 "
                         "gap. Only for comparing the two definitions on one "
                         "run; the flip-heavy cells are not trustworthy.")
    args = ap.parse_args()

    with open(args.dump, encoding="utf-8") as fh:
        payload = json.load(fh)
    cells = payload["cells"]
    cfg = payload["config"]

    print(f"cells: {', '.join(sorted(cells))}")
    print(f"config: {cfg}")

    # Which margin definition the dump carries decides whether any of the
    # fits below mean anything. The old dumps recorded the cell's *own* top-2
    # gap, which for a flipped position is the gap between a different pair of
    # candidates -- the regressand stops being the same variable as the
    # regressor, and the correlation falls apart exactly where flips are
    # common (0.738 at age 1, 0.198 at 'block'). Refuse rather than print
    # numbers that look fine and are not.
    mdef = payload.get("margin_def")
    if mdef != "signed_reference_ordering":
        print()
        print(f"this dump's margins are "
              f"'{mdef or 'own_top2_gap (legacy)'}', not "
              f"'signed_reference_ordering'.")
        print("Every fit below assumes one candidate ordering shared across "
              "cells. Re-dump with the current check_block_reuse.py, or pass "
              "--allow-legacy-margins to read it anyway and know the "
              "flip-heavy cells are contaminated.")
        if not args.allow_legacy_margins:
            return 1
        print("--allow-legacy-margins given: continuing on contaminated data.")

    if REFERENCE not in cells:
        print(f"\nno {REFERENCE} cell: there is nothing to regress against. "
              f"Include 16 in --bits and every_n:1 in --policies.")
        return 1

    names = sorted(cells)
    keys = joined(cells, names)
    print(f"\n{len(keys)} positions present in all {len(names)} cells")

    ref_m = [cells[REFERENCE][k][0] for k in keys]

    # Free cross-check, and the reason the signed form is worth a re-dump: a
    # negative margin *is* a flipped position, so the share of negatives must
    # reproduce the agreement row that check_block_reuse.py printed for the
    # same cell. If these two disagree, the dump and the sweep are not
    # describing the same run and nothing below is worth reading.
    print()
    print("=== 0. flips implied by the sign, against the printed rows ===")
    print(f"{'cell':>16} {'negative':>9} {'agree%':>8}")
    for name in names:
        col = [cells[name][k][0] for k in keys]
        neg = sum(1 for v in col if v < 0)
        print(f"{name:>16} {neg:>9} {100 * (1 - neg / len(col)):>8.2f}")
    ref_neg = sum(1 for v in ref_m if v < 0)
    if ref_neg:
        print()
        print(f"{ref_neg} negative margins in the reference cell "
              f"itself, which should be impossible: "
              f"there the committed token is the reference token "
              f"by construction. Check the canvas pinning.")

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

    # ---------------------------------------------------------------- 5
    kappa_table(cells, keys, names)
    return 0


def kappa_table(cells, keys, names) -> None:
    """The exact decomposition: where the sub-additivity actually lives.

    The shrinkage coefficient is a regression slope, so

        c = cov(m_ref, m_cell) / var(m_ref),      m_cell = m_ref + dm
        d = 1 - c = -cov(m_ref, dm) / var(m_ref)

    Covariance is bilinear. That makes ``d`` *strictly additive in the shift*:
    if the shifts add, the deficits add exactly -- no dimension argument, no
    concentration of measure, no fitted constant. So the whole interaction of
    the two errors collapses into one measurable quantity, the residual of that
    additivity projected back onto the reference margin:

        rho   = dm_both - dm_rounding - dm_staleness
        kappa = cov(m_ref, rho) / var(m_ref)
        d_both = d_r + d_s - kappa            <- an identity, not a model

    Which is why quadrature is a numerical coincidence rather than a law: it
    holds exactly when kappa happens to equal ``d_r + d_s - hypot(d_r, d_s)``,
    and nothing so far says it must.

    Two things worth watching. The two ways of computing kappa -- from the
    deficits and from the residual directly -- must agree to rounding; they are
    the same number and disagreement means the joining is wrong. And when the
    smaller error is much smaller, absorption in exact form would read
    ``kappa -> d_r``, since then ``d_both -> d_s`` and the smaller error adds
    nothing at all.
    """
    if REFERENCE not in cells:
        return
    ref = [cells[REFERENCE][k][0] for k in keys]
    n = len(ref)
    mref = sum(ref) / n
    var = sum((x - mref) ** 2 for x in ref) / n
    if var == 0:
        return

    def deficit(name):
        """1 - c, straight from the covariance rather than from a regression."""
        dm = [cells[name][k][0] - cells[REFERENCE][k][0] for k in keys]
        mdm = sum(dm) / n
        cov = sum((a - mref) * (b - mdm) for a, b in zip(ref, dm)) / n
        return -cov / var

    widths = sorted({x.split("/")[0] for x in names} - {"16"}, key=lambda b: -int(b))
    policies = [x.split("/", 1)[1] for x in names if x.startswith("16/")]
    policies = [p for p in policies if p != "every_n:1"]
    if not widths or not policies:
        return

    print("\n=== 5. exact decomposition: d_both = d_r + d_s - kappa ===")
    print()
    print(f"{'bits':>5} {'policy':>11} {'d_r':>7} {'d_s':>7} {'d_both':>7} "
          f"{'kappa':>7} {'direct':>7} {'k/d_r':>7} {'k needed':>9} {'diff':>7}")
    print("-" * 82)
    for policy in policies:
        stale = f"16/{policy}"
        if stale not in cells:
            continue
        d_s = deficit(stale)
        dm_s = [cells[stale][k][0] - cells[REFERENCE][k][0] for k in keys]
        for bits in widths:
            rnd, both = f"{bits}/every_n:1", f"{bits}/{policy}"
            if rnd not in cells or both not in cells:
                continue
            d_r, d_b = deficit(rnd), deficit(both)
            kappa = d_r + d_s - d_b

            # The same number the other way: straight from the residual.
            dm_r = [cells[rnd][k][0] - cells[REFERENCE][k][0] for k in keys]
            dm_b = [cells[both][k][0] - cells[REFERENCE][k][0] for k in keys]
            rho = [b - a - c for a, c, b in zip(dm_r, dm_s, dm_b)]
            mrho = sum(rho) / n
            direct = sum((a - mref) * (r - mrho) for a, r in zip(ref, rho)) / n / var

            needed = d_r + d_s - (d_r ** 2 + d_s ** 2) ** 0.5
            print(f"{bits:>5} {policy:>11} {d_r:>7.3f} {d_s:>7.3f} {d_b:>7.3f} "
                  f"{kappa:>7.3f} {direct:>7.3f} {kappa / d_r:>7.3f} "
                  f"{needed:>9.3f} {kappa - needed:>+7.3f}")
    print()
    print("  `kappa` and `direct` are the same quantity computed two ways and")
    print("  must agree. `k/d_r` at 1.0 is absorption in exact form: the")
    print("  smaller error adds nothing, d_both = d_s. `k needed` is what")
    print("  quadrature would require, and `diff` is how far the data is from")
    print("  needing it -- small there means quadrature fits, but it is a")
    print("  coincidence of this range, not a derivation.")


if __name__ == "__main__":
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
