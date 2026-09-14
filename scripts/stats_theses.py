"""Intervals and tests for the two theses that had none: К3 and К4.

Every answer-level thesis carries a McNemar p. The two that do not were read
off tables of means, and a ratio of means says nothing about how far it could
move. This script reads the dumps the probes write and puts a spread on them.
No GPU, no model: the expensive part is on the pod, this is arithmetic.

**К3, the axis cost.** ``check_key_error --dump`` writes the relative error of
every (canvas, layer) cell. The canvas is the independent unit -- layers of one
canvas share its text -- so the bootstrap resamples canvases and keeps the
layer set fixed, as the design fixes it. Three readings:

* one dump: the wrong-axis ratio for K and V, with a 95% percentile interval;
* two dumps of different models: the quotient of their ratios, resampled
  independently, and the share of draws where the quotient falls to one or
  below -- the bootstrap's one-sided p for "the MoE's axis costs more";
* two dumps of one model (``--paired``), QK-Norm on and skipped: the same
  canvases drawn for both, which is what makes the within-model leg one
  variable rather than two.

**К4, the rounding floor.** ``check_block_reuse --dump-margins`` writes the
signed decision margin of every committed position, in the reference's
ordering, so a position flipped exactly when its margin is negative. Flips are
counted per prompt, because positions of one prompt share a trajectory and are
not independent trials; a test on pooled positions would count 1024 trials
where there are eight. Two tests:

* paired by prompt: prompt *i* is the same text on both models (``text_ids``
  seeds the corpus draw, not the tokenizer), so the per-prompt rate
  difference is a paired quantity and the sign-flip test is exact;
* unpaired: prompts relabelled between models, for a reader who does not
  accept text as a pairing when the tokenizers differ.

    python scripts/stats_theses.py axis out/ke_l20_m50.json out/ke_l15_m50.json
    python scripts/stats_theses.py axis out/ke_l20_m50.json out/ke_l20_m50_nonorm.json --paired
    python scripts/stats_theses.py floor out/br_l20.json out/br_l15.json --cell 3/every_n:1
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from typing import Dict, List, Sequence, Tuple

Grid = List[List[float]]  # [canvas][layer]


# ---------------------------------------------------------------- primitives

def ratio_of_means(num: Grid, den: Grid, canvases: Sequence[int]) -> float:
    """Mean wrong-axis error over mean right-axis error, on these canvases.

    Ratio of means rather than mean of ratios, because that is what the table
    in RESULTS prints and the interval has to belong to the printed number.
    """
    n = sum(sum(num[c]) for c in canvases)
    d = sum(sum(den[c]) for c in canvases)
    return n / d


def percentile(draws: Sequence[float], q: float) -> float:
    s = sorted(draws)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def resample(n: int, rng: random.Random) -> List[int]:
    return [rng.randrange(n) for _ in range(n)]


def bootstrap_p_le_one(draws: Sequence[float]) -> float:
    """Share of draws at or below one, with the +1 that keeps it off zero."""
    return (1 + sum(1 for d in draws if d <= 1.0)) / (len(draws) + 1)


def sign_flip_p(diffs: Sequence[float], mc: int = 200_000,
                rng: random.Random = None) -> Tuple[float, bool]:
    """Two-sided paired permutation p on the mean difference.

    Exact when there are few enough pairs to enumerate every sign pattern,
    Monte Carlo past that. Returns (p, exact). The smallest exact p with n
    pairs is 2 / 2**n, so eight prompts cannot go below 0.0078 whatever the
    data say -- worth knowing before reading the number.
    """
    n = len(diffs)
    obs = abs(sum(diffs))
    if n <= 16:
        hits = total = 0
        for signs in itertools.product((1, -1), repeat=n):
            total += 1
            if abs(sum(s * d for s, d in zip(signs, diffs))) >= obs - 1e-12:
                hits += 1
        return hits / total, True
    rng = rng or random.Random(0)
    hits = 0
    for _ in range(mc):
        if abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (mc + 1), False


def relabel_p(a: Sequence[float], b: Sequence[float], mc: int = 200_000,
              rng: random.Random = None) -> Tuple[float, bool]:
    """Two-sided unpaired permutation p on the difference of mean rates."""
    pooled = list(a) + list(b)
    na, n = len(a), len(a) + len(b)
    obs = abs(sum(a) / na - sum(b) / len(b))
    tot = sum(pooled)

    def stat(idx) -> float:
        sa = sum(pooled[i] for i in idx)
        return abs(sa / na - (tot - sa) / (n - na))

    combos = 1
    for k in range(na):
        combos = combos * (n - k) // (k + 1)
    if combos <= 500_000:
        hits = sum(1 for idx in itertools.combinations(range(n), na)
                   if stat(idx) >= obs - 1e-12)
        return hits / combos, True
    rng = rng or random.Random(0)
    hits = sum(1 for _ in range(mc)
               if stat(rng.sample(range(n), na)) >= obs - 1e-12)
    return (hits + 1) / (mc + 1), False


# ---------------------------------------------------------------------- К3

def load_key(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    if d.get("shape") != "canvas x layer":
        raise SystemExit(f"{path}: not a check_key_error --dump file")
    return d


def axis_draws(d: dict, side: str, bits: int, g: int, B: int,
               rng: random.Random, canvases_list=None) -> Tuple[float, List[float]]:
    num = d["errors"][f"{side}/{bits}/channel/{g}"]   # scale per token: wrong for K
    den = d["errors"][f"{side}/{bits}/token/{g}"]     # scale per channel
    n = len(num)
    est = ratio_of_means(num, den, range(n))
    if canvases_list is None:
        canvases_list = [resample(n, rng) for _ in range(B)]
    return est, [ratio_of_means(num, den, c) for c in canvases_list]


def level_draws(d: dict, bits: int, g: int, canvases_list) -> Tuple[float, List[float]]:
    grid = d["errors"][f"K/{bits}/token/{g}"]
    n_l = len(grid[0])

    def m(c):
        return sum(sum(grid[i]) for i in c) / (len(c) * n_l)

    return m(range(len(grid))), [m(c) for c in canvases_list]


def ci(draws) -> str:
    return f"[{percentile(draws, 0.025):.2f}, {percentile(draws, 0.975):.2f}]"


def label(d: dict) -> str:
    c = d["config"]
    tag = c["model"].split("/")[-1]
    if c.get("skip_qk_norm"):
        tag += " (QK-Norm skipped)"
    return f"{tag}, mask {c['mask_ratio']:.2f}, {c['samples']} canvases"


def cmd_axis(args) -> int:
    a, b = load_key(args.a), load_key(args.b)
    rng = random.Random(args.seed)
    for d in (a, b):
        if args.bits not in d["config"]["bits"] or args.group not in d["config"]["group_size"]:
            raise SystemExit(f"{label(d)}: no {args.bits}-bit / group {args.group} cells")
    if args.paired:
        for key in ("model", "samples", "layers", "mask_ratio", "seq_len"):
            if a["config"][key] != b["config"][key]:
                raise SystemExit(
                    f"--paired needs the same {key} on both dumps "
                    f"({a['config'][key]} against {b['config'][key]}); without it "
                    "the shared canvas index names two different texts")

    n_a, n_b = len(a["errors"]["K/%d/token/%d" % (args.bits, args.group)]), \
        len(b["errors"]["K/%d/token/%d" % (args.bits, args.group)])
    idx_a = [resample(n_a, rng) for _ in range(args.boot)]
    idx_b = idx_a if args.paired else [resample(n_b, rng) for _ in range(args.boot)]

    print(f"{args.bits} bits, group {args.group}, {args.boot} bootstrap draws over "
          f"canvases, layers held fixed")
    print(f"ratio = error with one scale per token / error with one scale per "
          f"channel (the wrong axis for K over the right one)\n")

    res = {}
    for name, d, idx in (("A", a, idx_a), ("B", b, idx_b)):
        print(f"{name}: {label(d)}")
        for side in ("K", "V"):
            est, dr = axis_draws(d, side, args.bits, args.group, args.boot, rng, idx)
            res[(name, side)] = (est, dr)
            num = d["errors"][f"{side}/{args.bits}/channel/{args.group}"]
            den = d["errors"][f"{side}/{args.bits}/token/{args.group}"]
            per_layer = [sum(r[l] for r in num) / sum(r[l] for r in den)
                         for l in range(len(num[0]))]
            print(f"  {side} ratio {est:5.2f}x  95% CI {ci(dr)}   "
                  f"per layer {min(per_layer):.2f}-{max(per_layer):.2f}")
        est, dr = level_draws(d, args.bits, args.group, idx)
        res[(name, "level")] = (est, dr)
        print(f"  K error on the right axis {est:.3e}  95% CI "
              f"[{percentile(dr, .025):.3e}, {percentile(dr, .975):.3e}]")
        print()

    mode = "paired: the same canvases in every draw" if args.paired else \
        "independent resampling of the two dumps"
    print(f"A over B ({mode})")
    for side in ("K", "V"):
        est = res[("A", side)][0] / res[("B", side)][0]
        q = [x / y for x, y in zip(res[("A", side)][1], res[("B", side)][1])]
        print(f"  {side} ratio of ratios {est:5.2f}  95% CI {ci(q)}  "
              f"p(A <= B) = {bootstrap_p_le_one(q):.2g}")
    est = res[("A", "level")][0] / res[("B", "level")][0]
    q = [x / y for x, y in zip(res[("A", "level")][1], res[("B", "level")][1])]
    print(f"  K right-axis error, A / B {est:5.2f}  95% CI {ci(q)}  "
          f"p(A >= B) = {(1 + sum(1 for x in q if x >= 1)) / (len(q) + 1):.2g}")
    print("\n  Read: К3 holds if K's quotient stays clear of one while V's "
          "interval covers it -- the norm acts on K and not on V. The level "
          "line is К4's other half: a floor that is higher on the model whose "
          "K rounds *better* is not explained by the tensor.")
    return 0


# ---------------------------------------------------------------------- К4

def load_margins(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    if d.get("margin_def") != "signed_reference_ordering":
        raise SystemExit(
            f"{path}: margins are not signed in the reference's ordering, so a "
            "negative value does not mean a flip. Re-run check_block_reuse.")
    return d


def per_prompt(d: dict, cell: str) -> Tuple[List[int], List[int]]:
    if cell not in d["cells"]:
        raise SystemExit(f"cell {cell} not in dump; have {sorted(d['cells'])}")
    n = d["config"]["samples"]
    flips, total = [0] * n, [0] * n
    for key, vals in d["cells"][cell].items():
        i = int(key.split(":")[0])
        total[i] += 1
        flips[i] += vals[0] < 0
    return flips, total


def cmd_floor(args) -> int:
    a, b = load_margins(args.a), load_margins(args.b)
    rng = random.Random(args.seed)
    fa, ta = per_prompt(a, args.cell)
    fb, tb = per_prompt(b, args.cell)
    ra = [f / t for f, t in zip(fa, ta)]
    rb = [f / t for f, t in zip(fb, tb)]

    def pooled(f, t, idx):
        return sum(f[i] for i in idx) / max(sum(t[i] for i in idx), 1)

    print(f"cell {args.cell}: share of committed decisions the cache flipped, "
          f"text held to the reference\n")
    draws = {}
    for name, d, f, t, r in (("A", a, fa, ta, ra), ("B", b, fb, tb, rb)):
        idx = [resample(len(f), rng) for _ in range(args.boot)]
        dr = [pooled(f, t, i) for i in idx]
        draws[name] = (pooled(f, t, range(len(f))), dr)
        print(f"{name}: {d['config']['model'].split('/')[-1]}, "
              f"{len(f)} prompts, {sum(t)} decisions")
        print(f"  floor {100 * draws[name][0]:.2f}%  ({sum(f)} flips)  "
              f"95% CI [{100 * percentile(dr, .025):.2f}, "
              f"{100 * percentile(dr, .975):.2f}]%   "
              f"per prompt {', '.join(f'{100 * x:.1f}' for x in r)}")
    q = [x / y if y else float("inf") for x, y in zip(draws["A"][1], draws["B"][1])]
    est = draws["A"][0] / draws["B"][0] if draws["B"][0] else float("inf")
    finite = [x for x in q if x != float("inf")]
    print(f"\nA / B {est:.1f}x   95% CI [{percentile(finite, .025):.1f}, "
          f"{percentile(finite, .975):.1f}]x "
          f"({len(q) - len(finite)} draws with no flip on B, left out)")

    if len(ra) == len(rb) and a["config"].get("prompt_tokens") == b["config"].get("prompt_tokens"):
        p, exact = sign_flip_p([x - y for x, y in zip(ra, rb)], rng=rng)
        print(f"paired by prompt text, sign-flip on per-prompt rates: "
              f"p = {p:.2g} ({'exact' if exact else 'Monte Carlo'}; "
              f"floor with {len(ra)} pairs is {2 / 2 ** len(ra):.2g})")
    else:
        print("paired test skipped: prompt counts or lengths differ, so index i "
              "is not the same text on both")
    p, exact = relabel_p(ra, rb, rng=rng)
    print(f"unpaired, prompts relabelled between models: p = {p:.2g} "
          f"({'exact' if exact else 'Monte Carlo'})")
    print("\n  Both tests treat the prompt as the unit. A test on pooled "
          "positions would be far smaller and wrong: decisions of one "
          "trajectory are not independent trials.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ax = sub.add_parser("axis", help="К3: intervals on the axis ratio")
    ax.add_argument("a")
    ax.add_argument("b")
    ax.add_argument("--paired", action="store_true")
    ax.add_argument("--bits", type=int, default=4)
    ax.add_argument("--group", type=int, default=128)
    fl = sub.add_parser("floor", help="К4: the rounding floor between models")
    fl.add_argument("a")
    fl.add_argument("b")
    fl.add_argument("--cell", default="3/every_n:1")
    for p in (ax, fl):
        p.add_argument("--boot", type=int, default=10_000)
        p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    return cmd_axis(args) if args.cmd == "axis" else cmd_floor(args)


if __name__ == "__main__":
    raise SystemExit(main())
