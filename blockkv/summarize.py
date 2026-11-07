"""Tables from run_gsm8k outputs.

    python -m blockkv.summarize out/*.jsonl            # block -> accuracy, length, steps
    python -m blockkv.summarize --compare a.jsonl b.jsonl   # paired diff / determinism
"""
import argparse
import json
import math


def read(path):
    with open(path, encoding="utf-8") as f:
        return {r["idx"]: r for r in map(json.loads, filter(str.strip, f))}


def table(paths):
    print(f"{'file':40s} {'n':>5s} {'acc':>6s} {'±95':>5s} {'tokens':>7s} {'steps':>7s} {'tok/step':>8s} {'forced%':>7s}"
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
        print(f"{p[-40:]:40s} {n:5d} {100 * acc:6.1f} {100 * ci:5.1f} {tok:7.0f} {steps:7.0f} "
              f"{committed / max(sum(r['steps'] for r in rows), 1):8.2f} {100 * forced:7.1f}"
              f" {kv(rows, 'kv_bits_avg', '5.2f')} {kv(rows, 'kv_relerr_k', '7.4f')} {kv(rows, 'kv_relerr_v', '7.4f')}")


def kv(rows, key, fmt):
    """Mean of a quantization column over the rows that have it ('-' for uncompressed runs)."""
    vals = [r[key] for r in rows if key in r]
    width = int(fmt.split(".")[0])
    return format(sum(vals) / len(vals), fmt) if vals else "-".rjust(width)


def compare(a, b):
    ra, rb = read(a), read(b)
    common = sorted(ra.keys() & rb.keys())
    n = len(common)
    same_text = sum(ra[i]["text"] == rb[i]["text"] for i in common)
    only_a = sum(ra[i]["correct"] and not rb[i]["correct"] for i in common)
    only_b = sum(rb[i]["correct"] and not ra[i]["correct"] for i in common)
    acc_a = sum(ra[i]["correct"] for i in common) / n
    acc_b = sum(rb[i]["correct"] for i in common) / n
    # McNemar-style standard error of the paired difference.
    se = math.sqrt(only_a + only_b - (only_a - only_b) ** 2 / n) / n
    steps_a = sum(ra[i]["steps"] for i in common) / n
    steps_b = sum(rb[i]["steps"] for i in common) / n
    print(f"common problems: {n}, identical outputs: {same_text}")
    print(f"accuracy a={100 * acc_a:.1f} b={100 * acc_b:.1f} diff(b-a)={100 * (acc_b - acc_a):+.1f} ±{196 * se:.1f}")
    print(f"solved only by a: {only_a}, only by b: {only_b}")
    print(f"mean steps a={steps_a:.1f} b={steps_b:.1f} ({100 * (steps_b / steps_a - 1):+.1f}%)")


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
