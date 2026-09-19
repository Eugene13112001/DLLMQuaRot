"""Two open questions of the gain migration, read off one set of dumps.

1. Why QuaRot is not rescued. Rotation makes the key error isotropic, and isotropic
   error hurts the logits in proportion to the isotropic-noise factor
   sqrt(sum ||q||^2 ||k||^2) / sqrt(sum (q . k)^2). If that is the mechanism, QuaRot's
   logit error divided by the factor is the same number on every run -- LLaDA2.0-mini
   before and after the migration, and LLaDA-1.5.

2. What the migration leaves behind. After it the per-token cell sits at the edge of the
   collapse. Either the loudness moved into Q multiplies K's noise in those channels
   (then the error sits in the loud channels, and some smaller dose beats alpha = 1), or
   outliers the gain did not make remain (then it sits in the rest), or LLaDA2.0-mini is
   simply more fragile (then neither, and the per-layer crest already matches LLaDA-1.5).

    python scripts/diag_compare.py out/ke_d20_a0.json out/ke_d20_a025.json ... out/ke_d15.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

SCHEMES = (("per-channel", "token"), ("per-token", "channel"), ("QuaRot", "channel+rot"))


def mean(rows) -> float:
    vals = [v for row in rows for v in row]
    return sum(vals) / len(vals)


def per_layer(rows) -> List[float]:
    n = len(rows[0])
    return [sum(r[l] for r in rows) / len(rows) for l in range(n)]


def summarize(path: str, bits: int, group: int) -> Dict:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    cfg = d["config"]
    if not cfg.get("logit_error"):
        raise SystemExit(f"{path}: taken without --logit-error")
    out = {"name": os.path.basename(path).replace(".json", ""),
           "model": cfg["model"].split("/")[-1],
           "alpha": cfg.get("migrate_qk") or 0.0, "mode": cfg.get("migrate_mode") or "-"}
    for name, axis in SCHEMES:
        key = f"L/{bits}/{axis}/{group}"
        out[name] = mean(d["errors"][key]) if key in d["errors"] else None
        for part in ("Lloud", "Lrest"):
            pk = f"{part}/{bits}/{axis}/{group}"
            out[f"{name}:{part}"] = mean(d["errors"][pk]) if pk in d["errors"] else None
    out["cos"] = mean(d["qk_cos"]) if d.get("qk_cos") else None
    out["iso"] = mean(d["qk_iso"]) if d.get("qk_iso") else None
    out["crest"] = per_layer(d["crest"]["K"])
    return out


def fmt(x: Optional[float], w: int = 8, p: int = 3) -> str:
    return f"{x:>{w}.{p}f}" if x is not None else f"{'--':>{w}}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group", type=int, default=128)
    args = ap.parse_args()
    runs = [summarize(p, args.bits, args.group) for p in args.dumps]

    print(f"relative logit error at {args.bits} bits")
    print(f"{'run':<14}{'model':<16}{'alpha':>6}{'per-ch':>9}{'per-tok':>9}{'QuaRot':>9}"
          f"{'|cos|':>8}{'iso':>8}{'QuaRot/iso':>12}")
    for r in runs:
        ratio = r["QuaRot"] / r["iso"] if r["QuaRot"] and r["iso"] else None
        print(f"{r['name']:<14}{r['model']:<16}{r['alpha']:>6.2f}{fmt(r['per-channel'], 9)}"
              f"{fmt(r['per-token'], 9)}{fmt(r['QuaRot'], 9)}{fmt(r['cos'])}{fmt(r['iso'], 8, 2)}"
              f"{fmt(ratio, 12, 4)}")

    ratios = [r["QuaRot"] / r["iso"] for r in runs if r["QuaRot"] and r["iso"]]
    if len(ratios) > 1:
        spread = max(ratios) / min(ratios)
        print(f"\n1. QuaRot / isotropic factor ranges {min(ratios):.4f}..{max(ratios):.4f} "
              f"(max/min {spread:.2f}). Near 1 across runs whose QuaRot error differs by "
              f"far more means the factor explains QuaRot's error; far from 1 means it does not.")
        lq = [r["QuaRot"] for r in runs if r["QuaRot"]]
        print(f"   for scale: QuaRot's own error ranges max/min {max(lq) / min(lq):.2f}")

    print("\n2. where the per-token error sits (share of the logit error, loud channels / rest)")
    for r in runs:
        lo, re_ = r.get("per-token:Lloud"), r.get("per-token:Lrest")
        if lo is None:
            continue
        print(f"   {r['name']:<14} alpha {r['alpha']:.2f}  per-token loud {lo:.3f}  rest {re_:.3f}"
              f"   QuaRot loud {fmt(r.get('QuaRot:Lloud'), 6)}  rest {fmt(r.get('QuaRot:Lrest'), 6)}")
    doses = sorted([r for r in runs if r["model"] == runs[0]["model"] and r["per-token"]],
                   key=lambda r: (r["mode"] != "shrink" and r["alpha"] > 0, r["alpha"]))
    if doses:
        best = min(doses, key=lambda r: r["per-token"])
        print(f"   lowest per-token logit error on {runs[0]['model']}: {best['name']} "
              f"({best['per-token']:.3f}); alpha = 1 shrink gives "
              f"{next((fmt(r['per-token']) for r in doses if r['alpha'] == 1.0 and r['mode'] == 'shrink'), '--')}")

    print("\n3. key crest per layer (mean over canvases)")
    for r in runs:
        c = r["crest"]
        print(f"   {r['name']:<14} mean {sum(c) / len(c):5.2f}  max {max(c):5.2f}  "
              f"layers: {' '.join(f'{x:.1f}' for x in c)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
