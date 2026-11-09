"""RULER tables: every task counts the same, as in the benchmark's average.

    python -m blockkv.summarize_ruler out/ruler/16384

Pairs each `..._kv*.jsonl` of a folder with the uncompressed file of its block.
"""
import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

from blockkv.summarize import read


def task_mean(rows, key="score"):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r[key])
    return sum(sum(v) / len(v) for v in by_task.values()) / len(by_task)


def paired(ref, run):
    """(difference of the task-balanced scores, its 95% half-width), on the common problems."""
    by_task = defaultdict(list)
    for i in ref.keys() & run.keys():
        by_task[ref[i]["task"]].append(run[i]["score"] - ref[i]["score"])
    k = len(by_task)
    diff = sum(sum(d) / len(d) for d in by_task.values()) / k
    var = sum(_var(d) / len(d) for d in by_task.values()) / k ** 2
    return diff, 1.96 * math.sqrt(var)


def _var(d):
    m = sum(d) / len(d)
    return sum((x - m) ** 2 for x in d) / max(len(d) - 1, 1)


def block_of(path):
    return int(re.search(r"_b(\d+)", path.name).group(1))


def table(folder):
    files = sorted(Path(folder).glob("*.jsonl"), key=lambda p: (block_of(p), p.name))
    refs = {block_of(p): read(p) for p in files if "_kv" not in p.name}
    print(f"{'file':34s} {'n':>4s} {'prompt':>6s} {'score':>6s} {'diff':>6s} {'±95':>5s} {'steps':>6s} {'d.steps%':>8s}")
    for p in files:
        rows = read(p)
        if not rows:
            continue
        vals = list(rows.values())
        line = (f"{p.stem[-34:]:34s} {len(vals):4d} {median(r['prompt_tokens'] for r in vals):6.0f} "
                f"{100 * task_mean(vals):6.1f}")
        ref = refs.get(block_of(p))
        if ref and "_kv" in p.name and ref.keys() & rows.keys():
            common = ref.keys() & rows.keys()
            diff, ci = paired(ref, rows)
            a, b = (median(x[i]["steps"] for i in common) for x in (ref, rows))
            line += f" {100 * diff:+6.1f} {100 * ci:5.1f} {b:6.0f} {100 * (b / max(a, 1) - 1):+8.1f}"
        else:
            line += f" {'-':>6s} {'-':>5s} {median(r['steps'] for r in vals):6.0f} {'-':>8s}"
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    for folder in ap.parse_args().folders:
        print(f"== {folder}")
        table(folder)
