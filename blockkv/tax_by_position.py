"""Does the tax grow along the answer? Tokens per step by position in the answer, compressed vs exact.

    python -m blockkv.tax_by_position --cap 3072 REF::RUN [REF::RUN ...]
    python -m blockkv.tax_by_position --cap 3072 --edges 0 256 512 1024 2048 3072 \
        out/longproc/2k/sdar8b_b32.jsonl::out/longproc/2k/sdar8b_b32_kv2.jsonl

Each row's `per_step` lists the tokens committed at each step. Steps are put into answer segments by
the position of the first token they commit (--edges, in tokens). Per segment: tokens / steps of RUN
over the same of REF, on the clean pairs (neither answer a runaway, as in tax_decompose) where both
answers reach the segment, with a bootstrap 95% interval over problems. The input is the same for
both answers, so a ratio that falls along the answer means the tax grows with the answer itself.
"""
import argparse
import random

from blockkv.summarize import read
from blockkv.tax_decompose import repetitive


def segments(per_step, edges):
    """[(tokens, steps)] per segment [edges[k], edges[k+1])."""
    out = [[0, 0] for _ in range(len(edges) - 1)]
    pos = 0
    for n in per_step:
        for k in range(len(edges) - 1):
            if edges[k] <= pos < edges[k + 1]:
                out[k][0] += n
                out[k][1] += 1
                break
        pos += n
    return out


def by_position(ra, rb, cap, edges, repeat=None, reps=2000, seed=0):
    loop = lambda r: r["gen_tokens"] >= cap or (repeat is not None and repetitive(r.get("text"), **repeat))
    ids = [i for i in sorted(ra.keys() & rb.keys(), key=str) if not loop(ra[i]) and not loop(rb[i])]
    seg = {i: (segments(ra[i]["per_step"], edges), segments(rb[i]["per_step"], edges)) for i in ids}
    rows = []
    rng = random.Random(seed)
    for k in range(len(edges) - 1):
        use = [i for i in ids if seg[i][0][k][1] and seg[i][1][k][1]]
        def ratio(s):
            ta, sa = sum(seg[i][0][k][0] for i in s), sum(seg[i][0][k][1] for i in s)
            tb, sb = sum(seg[i][1][k][0] for i in s), sum(seg[i][1][k][1] for i in s)
            return (tb / sb) / (ta / sa) if sa and sb and ta else None
        r = ratio(use)
        ci = None
        if len(use) >= 2 and r is not None:
            vals = sorted(v for v in (ratio([rng.choice(use) for _ in use]) for _ in range(reps)) if v is not None)
            ci = vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]
        tps_a = (sum(seg[i][0][k][0] for i in use) / max(sum(seg[i][0][k][1] for i in use), 1)) if use else None
        rows.append(dict(lo=edges[k], hi=edges[k + 1], n=len(use), tps_ref=tps_a, ratio=r, ci=ci))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="REF::RUN")
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--edges", type=int, nargs="+", default=[0, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--no-repeat", action="store_true")
    ap.add_argument("--prompt-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                    help="only problems with MIN <= prompt_tokens < MAX")
    args = ap.parse_args(argv)
    repeat = None if args.no_repeat else dict(n=10, tail=600, min_distinct=0.3)
    for pair in args.pairs:
        ref, run = pair.split("::")
        name = run.replace("\\", "/").split("/")[-1].removesuffix(".jsonl")
        print(f"\n{name}")
        print(f"  {'segment':>11s} {'n':>4s} {'tok/step ref':>12s} {'run/ref':>8s} {'95%':>18s}")
        ra, rb = read(ref), read(run)
        if args.prompt_range:
            lo, hi = args.prompt_range
            ra = {i: x for i, x in ra.items() if lo <= x.get("prompt_tokens", -1) < hi}
        for r in by_position(ra, rb, args.cap, args.edges, repeat):
            ci = f"[{100 * (r['ci'][0] - 1):+.1f}; {100 * (r['ci'][1] - 1):+.1f}]" if r["ci"] else ""
            tps = f"{r['tps_ref']:.2f}" if r["tps_ref"] is not None else "-"
            rat = f"{r['ratio']:.3f}" if r["ratio"] is not None else "-"
            print(f"  {r['lo']:>5d}-{r['hi']:<5d} {r['n']:4d} {tps:>12s} {rat:>8s} {ci:>18s}")


if __name__ == "__main__":
    main()
