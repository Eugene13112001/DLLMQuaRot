"""Is the step increase a tax (fewer tokens per step) or runaway answers (longer answers)?

    python -m blockkv.tax_decompose --cap 512 REF::RUN [REF::RUN ...]
    python -m blockkv.tax_decompose --cap 2000 out/stage1/full/sdar8b_b32_t0.9.jsonl::out/stage1_kv/full/sdar8b_b32_t0.9_kv2.jsonl

On the problems both files share:
  steps x      = total steps of RUN / total steps of REF
  length x     = total answer tokens of RUN / of REF
  tok/step x   = (tokens / steps) of RUN / of REF
so steps x = length x / tok/step x exactly. `from length` is the share of log(steps x) that comes
from longer answers; the rest comes from fewer tokens per step (the tax proper).
An answer that reaches --cap tokens counts as a runaway (LongBench: 512, the answer limit; GSM8K:
2000). The `clean` columns repeat the numbers on the pairs where neither answer ran away, with a
paired bootstrap 95% interval for the clean step increase.
"""
import argparse
import math
import random

from blockkv.summarize import read


def tokens(r):
    return r["gen_tokens"]


def ratios(ra, rb, ids):
    sa, sb = sum(ra[i]["steps"] for i in ids), sum(rb[i]["steps"] for i in ids)
    la, lb = sum(tokens(ra[i]) for i in ids), sum(tokens(rb[i]) for i in ids)
    if not sa or not sb or not la or not lb:
        return None
    steps, length = sb / sa, lb / la
    return dict(steps=steps, length=length, tps=length / steps)


def bootstrap(ra, rb, ids, reps=2000, seed=0):
    """95% interval of the clean step increase (ratio - 1), resampling problems."""
    if len(ids) < 2:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        s = [rng.choice(ids) for _ in ids]
        sa = sum(ra[i]["steps"] for i in s)
        sb = sum(rb[i]["steps"] for i in s)
        vals.append(sb / sa - 1 if sa else 0.0)
    vals.sort()
    return vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1]


def decompose(ra, rb, cap):
    common = sorted(ra.keys() & rb.keys(), key=str)
    if not common:
        return None
    loop = lambda r: tokens(r) >= cap
    clean = [i for i in common if not loop(ra[i]) and not loop(rb[i])]
    allr, cl = ratios(ra, rb, common), ratios(ra, rb, clean)
    if allr is None:
        return None
    from_len = math.log(allr["length"]) / math.log(allr["steps"]) if abs(math.log(allr["steps"])) > 1e-9 else float("nan")
    return dict(n=len(common), loops_ref=sum(loop(ra[i]) for i in common), loops_run=sum(loop(rb[i]) for i in common),
                steps=allr["steps"], length=allr["length"], tps=allr["tps"], from_length=from_len,
                n_clean=len(clean), clean=cl, clean_ci=bootstrap(ra, rb, clean))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="REF::RUN")
    ap.add_argument("--cap", type=int, required=True, help="answer length counted as a runaway (LongBench 512, GSM8K 2000)")
    args = ap.parse_args(argv)
    print("| run | n | runaways ref→run | steps × | length × | tok/step × | share from length | clean n | clean steps × [95%] | clean tok/step × |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for pair in args.pairs:
        ref, run = pair.split("::")
        d = decompose(read(ref), read(run), args.cap)
        name = run.replace("\\", "/").split("/")[-1].removesuffix(".jsonl")
        if d is None:
            print(f"| {name} | – | | | | | | | | |")
            continue
        c = d["clean"] or {}
        ci = d["clean_ci"]
        ci_s = f" [{100 * ci[0]:+.1f}; {100 * ci[1]:+.1f}]" if ci else ""
        print(f"| {name} | {d['n']} | {d['loops_ref']}→{d['loops_run']} | {d['steps']:.3f} | {d['length']:.3f} | "
              f"{d['tps']:.3f} | {100 * d['from_length']:.0f}% | {d['n_clean']} | "
              f"{c.get('steps', float('nan')):.3f}{ci_s} | {c.get('tps', float('nan')):.3f} |")


if __name__ == "__main__":
    main()
