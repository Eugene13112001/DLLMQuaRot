"""Damage against refresh budget, in recomputed positions per step.

`check_block_reuse` reports damage against the *interval* -- "every_n:2", "block".
That is not a quantity. An interval of two steps means one thing at a block
length of 32 and another at 64, and another again when the sampler commits
three positions per step instead of one. Two of those are moving targets: the
block length has never been swept here, and DSB (2602.05992) argues a fixed
block is wrong in the first place.

The invariant is what a refresh actually costs:

    positions recomputed per step = block_length * (1 - hit_rate)

The hit rate is measured and printed by `check_block_reuse`, so this needs no
new run -- it re-plots rows that already exist onto an axis that survives a
change of block length or of parallelism.

What the axis is for: on it, "does the bit width shift the curve, or only move
a point along it" becomes answerable. If bits and refreshes were substitutes,
every width would lie on one frontier with one asymptote. They do not: at full
budget the exact cache reaches zero by construction while the quantized ones
stop at their rounding floor, which refreshing cannot buy down. Different
asymptotes, therefore different curves.

    python scripts/frontier.py
"""

from __future__ import annotations

BLOCK_LENGTH = 32

# Hit rate as printed by check_block_reuse -- 8 prompts, 1024 committed tokens
# per cell, K along tokens, group 128.
HIT_RATE = {
    "every_n:1": 0.000,
    "every_n:2": 0.500,
    "every_n:3": 0.656,
    "every_n:4": 0.750,
    "block": 0.969,
}

# Damage: share of committed tokens the cache changed, text held to the
# reference. The 16-bit / every_n:1 cell is zero by construction -- it is the
# reference, and carries no sampling error.
DAMAGE = {
    16: {"every_n:1": 0.0, "every_n:2": 5.0, "every_n:3": 8.4,
         "every_n:4": 11.4, "block": 62.3},
    4: {"every_n:1": 3.9, "every_n:2": 5.4, "every_n:3": 8.9,
        "every_n:4": 10.6, "block": 62.0},
    3: {"every_n:1": 5.6, "every_n:2": 7.9, "every_n:3": 9.8,
        "every_n:4": 12.5},
}

ORDER = ["every_n:1", "every_n:2", "every_n:3", "every_n:4", "block"]
WIDTHS = (16, 4, 3)


def cost(policy: str) -> float:
    """Positions whose K/V are recomputed per step, averaged over the block."""
    return BLOCK_LENGTH * (1.0 - HIT_RATE[policy])


def budget_for(bits: int, target: float) -> float | None:
    """Budget at which width ``bits`` reaches ``target`` damage.

    Linear between measured points, which is all four points support. Used only
    to state an exchange rate, never to claim a value between them.
    """
    pts = sorted((DAMAGE[bits][p], cost(p)) for p in ORDER if p in DAMAGE[bits])
    for (d0, c0), (d1, c1) in zip(pts, pts[1:]):
        if d0 <= target <= d1:
            return c0 if d1 == d0 else c0 + (target - d0) / (d1 - d0) * (c1 - c0)
    return None


def main() -> None:
    print("=== damage against recomputed positions per step ===\n")
    print(f"{'policy':>10} {'budget':>7}  " + "".join(f"{b:>8}b" for b in WIDTHS))
    print("-" * 44)
    for p in ORDER:
        cells = "".join(
            f"{DAMAGE[b][p]:>8.1f}" if p in DAMAGE[b] else f"{'-':>8}"
            for b in WIDTHS
        )
        print(f"{p:>10} {cost(p):>7.1f}  {cells}")

    print("\n=== the floor: what refreshing cannot buy down ===\n")
    for b in WIDTHS:
        print(f"  {b:>2} bit: {DAMAGE[b]['every_n:1']:>4.1f}% at full budget")
    print("\n  Different asymptotes, therefore not one curve. This is the "
          "shortest\n  statement of the result and it needs no statistics -- "
          "three numbers.")

    print("\n=== vertical read: damage attributable to the bits ===")
    print("   (width b minus 16 bit at the same budget)\n")
    print(f"{'budget':>7}  {'4 bit':>8} {'share':>7}   {'3 bit':>8} {'share':>7}")
    print("-" * 44)
    for p in ORDER:
        base = DAMAGE[16][p]
        line = f"{cost(p):>7.1f}  "
        for b in (4, 3):
            if p not in DAMAGE[b]:
                line += f"{'-':>8} {'-':>7}   "
                continue
            v = DAMAGE[b][p]
            share = 100.0 * (v - base) / v if v > 0 else 0.0
            line += f"{v - base:>+8.1f} {share:>6.0f}%   "
        print(line)
    print("\n  From 100% of the damage to none of it. The bit width is the "
          "whole\n  story at full budget and invisible at a quarter of it.")

    print("\n=== horizontal read: the exchange rate ===")
    print("   (budget needed to hold the same damage)\n")
    print(f"{'damage':>8}  " + "".join(f"{b:>9}b" for b in WIDTHS)
          + "   3-bit overhead")
    print("-" * 56)
    for target in (8.0, 10.0, 12.0):
        vals = {b: budget_for(b, target) for b in WIDTHS}
        line = f"{target:>7.1f}%  " + "".join(
            f"{vals[b]:>9.1f}" if vals[b] is not None else f"{'-':>9}"
            for b in WIDTHS
        )
        if vals[16] and vals[3]:
            line += f"   {100 * (vals[3] / vals[16] - 1):>+7.0f}%"
        print(line)
    print("\n  Memory is traded for compute, and the rate is not constant: it "
          "steepens\n  as the quality target tightens.")

    print("\nCaveats. The cost model charges a refresh for the whole block; an "
          "\nimplementation that refreshes only masked positions compresses the "
          "axis,\nmonotonically, so the ordering survives. The axis counts "
          "recomputation\nonly -- quantization has a compute cost and a memory "
          "saving of its own,\nand neither is on it.")


if __name__ == "__main__":
    main()
