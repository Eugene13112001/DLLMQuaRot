"""Is the step increase a tax (fewer tokens per step) or runaway answers (longer answers)?

    python -m blockkv.tax_decompose --cap 512 REF::RUN [REF::RUN ...]
    python -m blockkv.tax_decompose --cap 2000 out/stage1/full/sdar8b_b32_t0.9.jsonl::out/stage1_kv/full/sdar8b_b32_t0.9_kv2.jsonl

On the problems both files share:
  steps x      = total steps of RUN / total steps of REF
  length x     = total answer tokens of RUN / of REF
  tok/step x   = (tokens / steps) of RUN / of REF
so steps x = length x / tok/step x exactly. `from length` is the share of log(steps x) that comes
from longer answers; the rest comes from fewer tokens per step (the tax proper).
An answer counts as a runaway (a loop) if it reaches --cap tokens (LongBench: 512, the answer limit;
GSM8K: 2000; LongProc 8K: 9216) or if its text ends in a repetition: among the last --tail words,
the share of distinct --ngram-word n-grams is below --min-distinct (a loop that the model left before
the cap, or a long answer that only repeats itself). `in loops` is the share of the extra steps
(RUN minus REF) spent on the problems where either answer is a runaway. The `clean` columns repeat
the numbers on the pairs where neither answer ran away, with a paired bootstrap 95% interval for the
clean step increase.
"""
import argparse
import math
import random

from blockkv.summarize import read


def tokens(r):
    return r["gen_tokens"]


def repetitive(text, n=10, tail=600, min_distinct=0.3):
    """True if the last `tail` words of `text` are mostly repeated word n-grams."""
    words = (text or "").split()[-tail:]
    if len(words) < 4 * n:
        return False
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams) < min_distinct


def ratios(ra, rb, ids):
    sa, sb = sum(ra[i]["steps"] for i in ids), sum(rb[i]["steps"] for i in ids)
    la, lb = sum(tokens(ra[i]) for i in ids), sum(tokens(rb[i]) for i in ids)
    if not sa or not sb or not la or not lb:
        return None
    steps, length = sb / sa, lb / la
    return dict(steps=steps, length=length, tps=length / steps)


def bootstrap(ra, rb, ids, reps=2000, seed=0, what="steps"):
    """95% interval, resampling problems, of the step increase (what="steps": steps ratio - 1) or of
    the change in tokens per step (what="tps": tok/step ratio - 1)."""
    if len(ids) < 2:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        s = [rng.choice(ids) for _ in ids]
        r = ratios(ra, rb, s)
        vals.append(r[what] - 1 if r else 0.0)
    vals.sort()
    return vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1]


def decompose(ra, rb, cap, repeat=None):
    """repeat: None (cap only) or dict(n=, tail=, min_distinct=) for the text repetition test."""
    common = sorted(ra.keys() & rb.keys(), key=str)
    if not common:
        return None
    loop = lambda r: tokens(r) >= cap or (repeat is not None and repetitive(r.get("text"), **repeat))
    clean = [i for i in common if not loop(ra[i]) and not loop(rb[i])]
    wild = [i for i in common if loop(ra[i]) or loop(rb[i])]
    extra = sum(rb[i]["steps"] - ra[i]["steps"] for i in common)
    in_loops = sum(rb[i]["steps"] - ra[i]["steps"] for i in wild) / extra if extra else float("nan")
    allr, cl = ratios(ra, rb, common), ratios(ra, rb, clean)
    if allr is None:
        return None
    from_len = math.log(allr["length"]) / math.log(allr["steps"]) if abs(math.log(allr["steps"])) > 1e-9 else float("nan")
    return dict(n=len(common), loops_ref=sum(loop(ra[i]) for i in common), loops_run=sum(loop(rb[i]) for i in common),
                cap_ref=sum(tokens(ra[i]) >= cap for i in common), cap_run=sum(tokens(rb[i]) >= cap for i in common),
                in_loops=in_loops, steps=allr["steps"], length=allr["length"], tps=allr["tps"], from_length=from_len,
                n_clean=len(clean), clean=cl, clean_ci=bootstrap(ra, rb, clean),
                clean_tps_ci=bootstrap(ra, rb, clean, what="tps"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="REF::RUN")
    ap.add_argument("--cap", type=int, required=True,
                    help="answer length counted as a runaway (LongBench 512, GSM8K 2000, LongProc 8K 9216)")
    ap.add_argument("--tasks", nargs="+", help="only problems whose idx starts with one of these (e.g. tom_tracking travel_planning)")
    ap.add_argument("--prompt-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                    help="only problems with MIN <= prompt_tokens < MAX (the input length of the reference row)")
    ap.add_argument("--no-repeat", action="store_true", help="runaways by --cap only, no text repetition test")
    ap.add_argument("--ngram", type=int, default=10)
    ap.add_argument("--tail", type=int, default=600)
    ap.add_argument("--min-distinct", type=float, default=0.3)
    args = ap.parse_args(argv)
    repeat = None if args.no_repeat else dict(n=args.ngram, tail=args.tail, min_distinct=args.min_distinct)
    print("| run | n | runaways ref→run (at cap) | extra steps in loops | steps × | length × | tok/step × | share from length | clean n | clean steps × [95%] | clean tok/step × [95%] |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for pair in args.pairs:
        ref, run = pair.split("::")
        ra, rb = read(ref), read(run)
        if args.tasks:
            keep = lambda r: {i: x for i, x in r.items() if str(i).startswith(tuple(args.tasks))}
            ra, rb = keep(ra), keep(rb)
        if args.prompt_range:
            lo, hi = args.prompt_range
            ok = {i for i, x in ra.items() if lo <= x.get("prompt_tokens", -1) < hi}
            ra = {i: x for i, x in ra.items() if i in ok}
        d = decompose(ra, rb, args.cap, repeat)
        name = run.replace("\\", "/").split("/")[-1].removesuffix(".jsonl")
        if d is None:
            print(f"| {name} | – | | | | | | | | | |")
            continue
        c = d["clean"] or {}
        ci = d["clean_ci"]
        ci_s = f" [{100 * ci[0]:+.1f}; {100 * ci[1]:+.1f}]" if ci else ""
        tci = d["clean_tps_ci"]
        tci_s = f" [{100 * tci[0]:+.1f}; {100 * tci[1]:+.1f}]" if tci else ""
        print(f"| {name} | {d['n']} | {d['loops_ref']}→{d['loops_run']} ({d['cap_ref']}→{d['cap_run']}) | "
              f"{100 * d['in_loops']:.0f}% | {d['steps']:.3f} | {d['length']:.3f} | "
              f"{d['tps']:.3f} | {100 * d['from_length']:.0f}% | {d['n_clean']} | "
              f"{c.get('steps', float('nan')):.3f}{ci_s} | {c.get('tps', float('nan')):.3f}{tci_s} |")


if __name__ == "__main__":
    main()
