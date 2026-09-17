"""Why does the gain migration rescue the per-token key scale and not QuaRot?

Reads check_key_error --logit-error dumps and puts the relative error of the attention
logits, ||q (K - Q(K))^T|| / ||q K^T||, next to the answers each cell produced. Three
hypotheses make different predictions about the same four numbers at two bits:

  H1  length, not peaks. Rotation spreads the quantization noise evenly over the head, so
      its size follows the length of the key, and after back-rotation it lands on the
      channels where the migration made q loud. The migration shrinks the peak of K a lot
      and its length little, so the per-token scale gains and QuaRot does not:
          L(quarot, migrated) >= L(quarot, base)
  H2  helped, but not enough. The migration lowers QuaRot's logit error too, just not
      below the level at which this model stops collapsing:
          L(quarot, migrated) < L(quarot, base), and
          L(quarot, migrated) > L(per-token, migrated)   (the rescued cell, 57.5%)
  H3  this model is more fragile, not QuaRot worse. LLaDA-1.5 survives QuaRot at two bits
      (84.0%); if its QuaRot logit error is no lower than LLaDA2.0-mini's after the
      migration, the same error is tolerated by one model and fatal to the other -- the
      three-times-higher rounding floor of section 5 -- and the gain is not the story.

Before any of that is read, the proxy has to order the cells whose answers are known:
per-channel (84.5) < per-token migrated (57.5) < per-token base and QuaRot (1.5). If it
does not, logit error is not what decides this collapse and none of the three verdicts
below should be trusted.

    python scripts/logit_compare.py --base out/ke_l20_base_L.json --mig out/ke_l20_mig_L.json \\
        [--other out/ke_l15_base_L.json] --bits 2
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Optional

# axis names are grouping directions: "token" = one scale per channel, "channel" = per token
CELLS = (("per-channel", "token"), ("per-token", "channel"), ("QuaRot", "channel+rot"))


def load(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    if not d.get("config", {}).get("logit_error"):
        raise SystemExit(f"{path}: taken without --logit-error, it has no logit cells")
    return d


def logit_errors(d: dict, bits: int, group: int) -> Dict[str, float]:
    out = {}
    for name, axis in CELLS:
        key = f"L/{bits}/{axis}/{group}"
        if key not in d["errors"]:
            raise SystemExit(f"missing cell {key}; was the run taken with --rotate?")
        rows = d["errors"][key]
        vals = [v for row in rows for v in row]
        out[name] = sum(vals) / len(vals)
    return out


def verdicts(base: Dict[str, float], mig: Dict[str, float],
             other: Optional[Dict[str, float]]) -> Dict[str, Optional[bool]]:
    """Which predictions hold. None where the needed number is missing."""
    ordered = (mig["per-channel"] < mig["per-token"]
               and mig["per-token"] < base["per-token"]
               and mig["per-token"] < base["QuaRot"])
    h1 = mig["QuaRot"] >= base["QuaRot"]
    h2 = (mig["QuaRot"] < base["QuaRot"]) and (mig["QuaRot"] > mig["per-token"])
    h3 = None if other is None else other["QuaRot"] >= mig["QuaRot"]
    return {"proxy orders the known answers": ordered, "H1": h1, "H2": h2, "H3": h3}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--mig", required=True)
    ap.add_argument("--other", default=None, help="LLaDA-1.5 dump, for H3")
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group", type=int, default=128)
    args = ap.parse_args()

    b = logit_errors(load(args.base), args.bits, args.group)
    m = logit_errors(load(args.mig), args.bits, args.group)
    o = logit_errors(load(args.other), args.bits, args.group) if args.other else None

    answers = {"per-channel": ("84.5", "84.5"), "per-token": ("1.5", "57.5"),
               "QuaRot": ("1.5", "0.0")}
    print(f"relative logit error at {args.bits} bits, LLaDA2.0-mini (answers at 2 bits in brackets)")
    print(f"{'':>12} {'base':>18} {'migrated':>18} {'mig/base':>9}"
          + (f" {'LLaDA-1.5':>10}" if o else ""))
    for name, _ in CELLS:
        ab, am = answers[name]
        line = (f"{name:>12} {b[name]:>10.3e} ({ab:>4}) {m[name]:>10.3e} ({am:>4}) "
                f"{m[name] / b[name]:>8.2f}x")
        if o:
            line += f" {o[name]:>10.3e}"
        print(line)

    print()
    for k, v in verdicts(b, m, o).items():
        mark = "n/a" if v is None else ("holds" if v else "fails")
        print(f"  {k:<32} {mark}")
    if args.bits != 2:
        print("  (the answers in brackets are two-bit answers; at other widths read the "
              "ordering only as a tensor statement)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
