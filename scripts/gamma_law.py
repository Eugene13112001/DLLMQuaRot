"""Does the K-norm gain predict, layer by layer, how much the wrong key axis costs?

The QK-Norm finding stands on two models and a probe that reads keys with the norm skipped. What would make it a
law is a prediction from the weights alone: the norm multiplies every channel of every key by a fixed gain
gamma, so a layer whose gains are spread far apart should hold a few loud fixed channels, and in that layer a
scale per token should cost much more than a scale per channel. That is checkable without a sampler.

Inputs are ``check_key_error --dump`` files taken with ``--all-layers --split-rope`` (and ``--rotate`` for the
QuaRot column):

    python scripts/gamma_law.py out/law_l20.json --nonorm out/law_l20_nonorm.json

For each layer it reads the gain vector of the K norm and the measured costs, and reports Spearman rank
correlations across layers with a permutation p-value and a bootstrap interval over layers.

Two controls are built in.

* **RoPE.** LLaDA2.0-mini rotates only part of each head. Rotary mixes channel i with channel i + rd/2 at a
  position-dependent angle, so in the rotated part the relevant gain is the pair's combined gain; in the
  unrotated part it is the channel's own gain and rotary cannot contribute anything. The law has to hold in the
  unrotated part, or it is not about the norm.
* **The layer's own key structure.** With ``--nonorm`` the same keys read before the norm give each layer's cost
  without the gain; the ratio of the two isolates what the gain adds. That is the quantity the gain spread should
  predict most directly.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple


# ------------------------------------------------------------------ statistics

def ranks(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if sa == 0 or sb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    return pearson(ranks(a), ranks(b))


def perm_p(a: Sequence[float], b: Sequence[float], n: int = 10_000, seed: int = 0) -> float:
    """Two-sided permutation p for Spearman's rho, shuffling layers."""
    rng = random.Random(seed)
    obs = abs(spearman(a, b))
    if math.isnan(obs):
        return float("nan")      # a constant feature or measure has no rank order to test
    b = list(b)
    hits = 0
    for _ in range(n):
        rng.shuffle(b)
        if abs(spearman(a, b)) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n + 1)


def boot_ci(a: Sequence[float], b: Sequence[float], n: int = 5_000, seed: int = 1) -> Tuple[float, float]:
    rng = random.Random(seed)
    m = len(a)
    draws = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        r = spearman([a[i] for i in idx], [b[i] for i in idx])
        if not math.isnan(r):
            draws.append(r)
    draws.sort()
    if not draws:
        return float("nan"), float("nan")
    return draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]


# ------------------------------------------------------------------ gain features

def pair_gains(g: Sequence[float], rd: int) -> List[float]:
    """Combined gain of each rotary pair (i, i + rd/2): rotary spreads a channel's energy over its pair."""
    half = rd // 2
    return [math.sqrt(g[i] ** 2 + g[i + half] ** 2) for i in range(half)]


def features(g: Sequence[float]) -> Dict[str, float]:
    a = sorted(abs(x) for x in g)
    n = len(a)
    if n == 0:
        return {}
    med = a[n // 2] if n % 2 else (a[n // 2 - 1] + a[n // 2]) / 2
    mean = sum(a) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in a) / n)
    energy = sum(x * x for x in a)
    top = sum(x * x for x in a[-max(1, n // 16):])
    return {
        "max/median": a[-1] / med if med else float("inf"),
        "cv": std / mean if mean else float("nan"),
        "top-share": top / energy if energy else float("nan"),
    }


# ------------------------------------------------------------------ dump reading

def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    if d.get("shape") != "canvas x layer":
        raise SystemExit(f"{path}: not a check_key_error --dump file")
    return d


def layer_ratio(d: dict, num: str, den: str) -> List[float]:
    """Per-layer ratio of summed errors, canvases pooled."""
    try:
        gn, gd = d["errors"][num], d["errors"][den]
    except KeyError as exc:
        raise SystemExit(f"missing cells {exc} -- rerun check_key_error with the needed flags") from exc
    n_l = len(gn[0])
    return [sum(r[l] for r in gn) / sum(r[l] for r in gd) for l in range(n_l)]


def cell(side: str, bits: int, axis: str, group: int, part: Optional[str] = None) -> str:
    return f"{side}/{bits}/{axis}{'@' + part if part else ''}/{group}"


def measures(d: dict, bits: int, group: int, part: Optional[str]) -> Dict[str, List[float]]:
    out = {"axis": layer_ratio(d, cell("K", bits, "channel", group, part), cell("K", bits, "token", group, part))}
    if cell("K", bits, "channel+rot", group, part) in d["errors"]:
        out["quarot"] = layer_ratio(d, cell("K", bits, "channel+rot", group, part), cell("K", bits, "token", group, part))
    return out


def run(args) -> int:
    d = load(args.dump)
    cfg = d["config"]
    layers = cfg["layers"]
    gam = d.get("gamma", {}).get("k") or []
    if not gam or any(g is None for g in gam):
        raise SystemExit("the dump has no K-norm gains for every layer -- this model has no QK-Norm, or the dump "
                         "predates the gain recording; nothing to correlate")
    if cfg.get("skip_qk_norm"):
        raise SystemExit("the main dump must be the one WITH the norm; pass the skipped one as --nonorm")
    rds = d.get("rotary_dim") or [len(gam[0])] * len(layers)
    applied = d.get("rope_applied") or []
    if applied and not all(applied):
        print("WARNING: the probe did not apply RoPE on some layers -- the rotated/unrotated split is meaningless there")
    head = len(gam[0])
    nn = load(args.nonorm) if args.nonorm else None
    if nn is not None and nn["config"]["layers"] != layers:
        raise SystemExit("--nonorm dump covers different layers")

    parts: List[Tuple[Optional[str], str]] = [(None, "whole head")]
    if cfg.get("split_rope"):
        parts.append(("rot", "part RoPE rotates"))
        if any(rd < head for rd in rds):
            parts.append(("pass", "part RoPE leaves alone"))

    print(f"{cfg['model']}: {len(layers)} layers, {cfg['samples']} canvases, mask {cfg['mask_ratio']:.2f}, "
          f"{args.bits} bits, group {args.group}, head {head}, rotary {sorted(set(rds))}")
    print("gain features per layer: max/median of |gamma|, coefficient of variation, share of gain energy in the "
          "top 1/16 of channels\n")

    for part, label in parts:
        feats = []
        for g, rd in zip(gam, rds):
            if part == "rot":
                feats.append(features(pair_gains(g, rd)))
            elif part == "pass":
                feats.append(features(g[rd:]))
            else:
                feats.append(features(g))
        meas = measures(d, args.bits, args.group, part)
        if nn is not None:
            base = measures(nn, args.bits, args.group, part)
            meas["axis added by norm"] = [a / b for a, b in zip(meas["axis"], base["axis"])]
            if "quarot" in meas and "quarot" in base:
                meas["quarot added by norm"] = [a / b for a, b in zip(meas["quarot"], base["quarot"])]

        print(f"=== {label} ===")
        cols = list(meas)
        print("layer  " + "  ".join(f"{c:>20}" for c in cols) + "  " + "  ".join(f"{f:>11}" for f in feats[0]))
        for i, li in enumerate(layers):
            print(f"{li:>5}  " + "  ".join(f"{meas[c][i]:>19.2f}x" for c in cols) + "  "
                  + "  ".join(f"{feats[i][f]:>11.3f}" for f in feats[0]))
        print()
        for c in cols:
            for f in feats[0]:
                xs = [ft[f] for ft in feats]
                ys = meas[c]
                rho = spearman(xs, ys)
                lo, hi = boot_ci(xs, ys, n=args.boot)
                p = perm_p(xs, ys, n=args.perm)
                print(f"  rho({f:>10} , {c:<20}) = {rho:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  permutation p = {p:.3g}")
        print()

    print("Read: the law needs a clear positive rho in the part RoPE leaves alone, and most of all against "
          "'axis added by norm'. A rho that appears only in the rotated part points at rotary, not at the gain. "
          f"With {len(layers)} layers the smallest permutation p is about 1/{args.perm}; a bootstrap over layers "
          "treats layers as exchangeable, which neighbouring layers are not quite.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("--nonorm", default=None, help="the same run with --skip-qk-norm")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--boot", type=int, default=5_000)
    ap.add_argument("--perm", type=int, default=10_000)
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
