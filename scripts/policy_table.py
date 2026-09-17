"""Price of cache bits against the refresh policy, on answers.

Reads the cells of scripts/pod/policy.sh and prints, for every policy, the accuracy at each
width and the price of the width against 16 bits at the *same* policy, with an exact
McNemar test on the paired questions. The question section 2.9 asked on decisions is
whether that price shrinks as the cache goes staler; this is the same question on answers.

    python scripts/policy_table.py --dir out
"""

from __future__ import annotations

import argparse
import json
import os
from math import comb
from typing import Dict, List, Optional, Sequence, Tuple

POLICIES = (("n1", "every step"), ("n2", "every 2"), ("n4", "every 4"), ("blk", "once per block"))
WIDTHS = (16, 3, 2)


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> Tuple[int, int, float]:
    """Questions b gains over a, questions b loses, exact two-sided p."""
    if len(a) != len(b):
        raise ValueError("the two runs cover different numbers of questions")
    up = sum(1 for x, y in zip(a, b) if not x and y)
    down = sum(1 for x, y in zip(a, b) if x and not y)
    n = up + down
    if n == 0:
        return up, down, 1.0
    tail = sum(comb(n, k) for k in range(min(up, down) + 1)) / 2 ** n
    return up, down, min(1.0, 2 * tail)


def load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def correct(run: dict) -> List[bool]:
    return [bool(s["correct"]) for s in run["samples"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="out")
    ap.add_argument("--prefix", default="l20_P")
    args = ap.parse_args()

    runs: Dict[Tuple[int, str], dict] = {}
    for w in WIDTHS:
        for key, _ in POLICIES:
            r = load(os.path.join(args.dir, f"{args.prefix}{w}_{key}.json"))
            if r is not None:
                runs[(w, key)] = r

    print(f"{'policy':>16} " + " ".join(f"{str(w) + ' bit':>9}" for w in WIDTHS))
    for key, label in POLICIES:
        cells = []
        for w in WIDTHS:
            r = runs.get((w, key))
            cells.append(f"{r['accuracy'] * 100:>9.1f}" if r else f"{'--':>9}")
        print(f"{label:>16} " + " ".join(cells))

    print("\nprice of the width against 16 bits at the same policy "
          "(points; questions lost / gained; McNemar p)")
    for key, label in POLICIES:
        base = runs.get((16, key))
        if base is None:
            continue
        parts = []
        for w in WIDTHS[1:]:
            r = runs.get((w, key))
            if r is None:
                continue
            gained, lost, p = mcnemar(correct(base), correct(r))
            price = (base["accuracy"] - r["accuracy"]) * 100
            parts.append(f"{w} bit: {price:+5.1f}  ({lost} lost / {gained} gained, p = {p:.2g})")
        if parts:
            print(f"{label:>16}  " + "   ".join(parts))

    print("\nprice of staleness at 16 bits against every step")
    ref = runs.get((16, "n1"))
    if ref is not None:
        for key, label in POLICIES[1:]:
            r = runs.get((16, key))
            if r is None:
                continue
            gained, lost, p = mcnemar(correct(ref), correct(r))
            print(f"{label:>16}  {(ref['accuracy'] - r['accuracy']) * 100:+5.1f}  "
                  f"({lost} lost / {gained} gained, p = {p:.2g})")

    print("\nRead: on decisions (2.9) the price of 3 bits fell from 5.6% at every step to "
          "noise at every 4. The same shape here means the law reaches answers; a flat "
          "price means it stops at decisions. 200 questions resolve about 5 points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
