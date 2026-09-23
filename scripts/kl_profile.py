"""Is the movement that matters the average one, or the worst layer?

Three Fast-dLLM-v2 cells sit at the same mean KL -- 0.348, 0.351, 0.355 -- and score 43.5,
1.5 and 28.0, so the mean over layers and canvases is not what accuracy follows. The mean
hides shape: a rotation spreads its error over every channel of every layer, while a
two-bit per-channel scale with the bias taken out may leave a few layers far worse than
the rest, and one ruined layer is enough to ruin what reads from it.

This prints the per-layer profile of any recorded cell and the order statistics that could
replace the mean -- max, the top three layers, the 90th percentile -- so the candidate can
be checked against the accuracies rather than assumed.

    python scripts/kl_profile.py out/ke_afd.json:4:channel+rot \
        out/ke_apb.json:2:token out/ke_amean.json:2:token --metric KL
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List


def cell(spec: str, metric: str, group: int):
    # rsplit: a Windows path carries a colon of its own ("C:\...").
    path, bits, axis = spec.rsplit(":", 2)
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    key = f"{metric}/{bits}/{axis}/{group}"
    rows = d["errors"].get(key)
    if rows is None:
        raise SystemExit(f"{path}: no cell {key}; it has "
                         f"{sorted(k for k in d['errors'] if k.startswith(metric))[:4]}")
    layers = d["config"]["layers"]
    per_layer = [sum(r[i] for r in rows) / len(rows) for i in range(len(layers))]
    return {"name": f"{os.path.basename(path)[:-5]} {bits}b {axis}",
            "layers": layers, "per_layer": per_layer}


def pct(v: List[float], q: float) -> float:
    x = sorted(v)
    i = min(len(x) - 1, int(round(q * (len(x) - 1))))
    return x[i]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cells", nargs="+", help="dump.json:bits:axis, axis as recorded "
                                             "(token, channel, channel+rot, ...)")
    ap.add_argument("--metric", default="KL", help="KL, Lc, Flip, Eff, K, V")
    ap.add_argument("--group", type=int, default=128)
    args = ap.parse_args()

    runs = [cell(c, args.metric, args.group) for c in args.cells]
    print(f"{args.metric} per layer, mean over canvases")
    print(f"{'cell':<34}{'mean':>8}{'max':>8}{'top3':>8}{'p90':>8}{'max/mean':>10}")
    for r in runs:
        v = r["per_layer"]
        m = sum(v) / len(v)
        top3 = sum(sorted(v)[-3:]) / 3
        print(f"{r['name']:<34}{m:>8.3f}{max(v):>8.3f}{top3:>8.3f}{pct(v, 0.9):>8.3f}"
              f"{max(v) / m if m else 0:>10.2f}")
    print()
    for r in runs:
        v = r["per_layer"]
        worst = sorted(range(len(v)), key=lambda i: -v[i])[:3]
        print(f"{r['name']:<34}worst layers "
              + ", ".join(f"{r['layers'][i]} ({v[i]:.3f})" for i in worst))
        print(f"{'':<34}" + " ".join(f"{x:.2f}" for x in v))
    print()
    print("  If the accuracies follow one of these columns where they do not follow the")
    print("  mean, that column is the candidate. If none of them orders the cells, the")
    print("  difference is not in how far attention moves on a probe canvas at all, and")
    print("  the place to look next is the generation path -- a cache error is written")
    print("  once and read by every later block, and the probe sees one canvas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
