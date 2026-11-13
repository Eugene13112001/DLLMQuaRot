"""Tables from run_gsm8k outputs.

    python -m blockkv.summarize out/*.jsonl            # block -> accuracy, length, steps
    python -m blockkv.summarize --compare a.jsonl b.jsonl   # paired diff / determinism
"""
import argparse
import json
import math
from statistics import median

RUNAWAY = 2000   # tokens: such answers ran into the length limit, and they swamp the mean


def read(path):
    with open(path, encoding="utf-8") as f:
        return {r["idx"]: r for r in map(json.loads, filter(str.strip, f))}


def table(paths):
    print(f"{'file':40s} {'n':>5s} {'acc':>6s} {'±95':>5s} {'tokens':>7s} {'steps':>7s} {'med.st':>7s} {'runaw':>5s} {'tok/step':>8s} {'forced%':>7s}"
          f" {'bits':>5s} {'errK':>7s} {'errV':>7s}")
    for p in paths:
        rows = list(read(p).values())
        n = len(rows)
        if not n:
            continue
        acc = sum(r["correct"] for r in rows) / n
        ci = 1.96 * math.sqrt(acc * (1 - acc) / n)
        tok = sum(r["gen_tokens"] for r in rows) / n
        steps = sum(r["steps"] for r in rows) / n
        committed = sum(sum(r["per_step"]) for r in rows)
        forced = sum(r["forced"] for r in rows) / max(committed, 1)
        print(f"{p[-40:]:40s} {n:5d} {100 * acc:6.1f} {100 * ci:5.1f} {tok:7.0f} {steps:7.0f} {median(r['steps'] for r in rows):7.0f} {runaways(rows):5d} "
              f"{committed / max(sum(r['steps'] for r in rows), 1):8.2f} {100 * forced:7.1f}"
              f" {kv(rows, 'kv_bits_avg', '5.2f')} {kv(rows, 'kv_relerr_k', '7.4f')} {kv(rows, 'kv_relerr_v', '7.4f')}")


def runaways(rows):
    return sum(r["gen_tokens"] >= RUNAWAY for r in rows)


def change(key, ra, rb, common):
    """One comparison line for `key`: medians, then means without the runaway answers."""
    calm = [i for i in common if ra[i]["gen_tokens"] < RUNAWAY and rb[i]["gen_tokens"] < RUNAWAY]
    mid_a, mid_b = (median(r[i][key] for i in common) for r in (ra, rb))
    mean_a, mean_b = (sum(r[i][key] for i in calm) / max(len(calm), 1) for r in (ra, rb))
    print(f"{key}: median a={mid_a:.0f} b={mid_b:.0f} ({100 * (mid_b / mid_a - 1):+.1f}%); "
          f"mean without runaways a={mean_a:.1f} b={mean_b:.1f} ({100 * (mean_b / mean_a - 1):+.1f}%, n={len(calm)})")


def kv(rows, key, fmt):
    """Mean of a quantization column over the rows that have it ('-' for uncompressed runs)."""
    vals = [r[key] for r in rows if key in r]
    width = int(fmt.split(".")[0])
    return format(sum(vals) / len(vals), fmt) if vals else "-".rjust(width)


def committed(r):
    """Tokens committed in this answer's blocks (whole blocks, not only up to the stop token)."""
    return sum(r["per_step"]) if "per_step" in r else r["gen_tokens"]


def paired(ra, rb):
    """Paired numbers of run b against run a over their common problems."""
    common = sorted(ra.keys() & rb.keys())
    n = len(common)
    calm = [i for i in common if ra[i]["gen_tokens"] < RUNAWAY and rb[i]["gen_tokens"] < RUNAWAY]
    wild = [i for i in common if i not in set(calm)]
    only_a = sum(ra[i]["correct"] and not rb[i]["correct"] for i in common)
    only_b = sum(rb[i]["correct"] and not ra[i]["correct"] for i in common)
    out = dict(n=n, calm=len(calm), same_text=sum(ra[i]["text"] == rb[i]["text"] for i in common),
               only_a=only_a, only_b=only_b,
               acc_a=sum(ra[i]["correct"] for i in common) / n, acc_b=sum(rb[i]["correct"] for i in common) / n,
               # McNemar-style standard error of the paired difference.
               se=math.sqrt(only_a + only_b - (only_a - only_b) ** 2 / n) / n,
               runaway_a=runaways([ra[i] for i in common]), runaway_b=runaways([rb[i] for i in common]))
    sa, sb = (sum(r[i]["steps"] for i in common) for r in (ra, rb))
    if sa and sb:                # the batched runner does not count steps
        ca, cb = (sum(r[i]["steps"] for i in calm) for r in (ra, rb))
        out.update(
            steps_a=sa / n, steps_b=sb / n,
            calm_steps_a=ca / max(len(calm), 1), calm_steps_b=cb / max(len(calm), 1),
            med_steps_a=median(ra[i]["steps"] for i in common), med_steps_b=median(rb[i]["steps"] for i in common),
            # parallelism: committed tokens per step, without runaway answers
            tps_a=sum(committed(ra[i]) for i in calm) / max(ca, 1),
            tps_b=sum(committed(rb[i]) for i in calm) / max(cb, 1),
            # extra steps per problem, and the part of them spent in runaway answers
            extra=(sb - sa) / n,
            extra_runaway=sum(rb[i]["steps"] - ra[i]["steps"] for i in wild) / n)
    return out


def compare(a, b):
    ra, rb = read(a), read(b)
    common = sorted(ra.keys() & rb.keys())
    p = paired(ra, rb)
    print(f"common problems: {p['n']}, identical outputs: {p['same_text']}")
    print(f"accuracy a={100 * p['acc_a']:.1f} b={100 * p['acc_b']:.1f} diff(b-a)={100 * (p['acc_b'] - p['acc_a']):+.1f} ±{196 * p['se']:.1f}")
    print(f"solved only by a: {p['only_a']}, only by b: {p['only_b']}")
    print(f"runaway answers a={p['runaway_a']} b={p['runaway_b']}")
    change("gen_tokens", ra, rb, common)
    if "steps_a" in p:
        print(f"mean steps a={p['steps_a']:.1f} b={p['steps_b']:.1f} ({100 * (p['steps_b'] / p['steps_a'] - 1):+.1f}%)")
        change("steps", ra, rb, common)
        print(f"tokens per step without runaways a={p['tps_a']:.2f} b={p['tps_b']:.2f} "
              f"({p['tps_b'] - p['tps_a']:+.2f}, {100 * (p['tps_b'] / p['tps_a'] - 1):+.1f}%)")
        share = p["extra_runaway"] / p["extra"] if p["extra"] else float("nan")
        print(f"extra steps per problem {p['extra']:+.2f}, of them in runaway answers {p['extra_runaway']:+.2f} ({100 * share:.0f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        assert len(args.files) == 2
        compare(*args.files)
    else:
        table(args.files)
