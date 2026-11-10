"""DECAL's matched commit temperature from runs at several temperatures.

    python -m blockkv.tmatch REF.jsonl RUN_T1.0.jsonl RUN_T0.95.jsonl ...

T_match is the largest temperature at which the compressed run needs no more steps
than the uncompressed reference, linearly interpolated between the measured ones.
Steps are averaged over the problems common to all files, without runaway answers.
"""
import argparse
import json
from pathlib import Path

from blockkv.summarize import RUNAWAY, read


def temperature(path):
    meta = Path(path).with_suffix(".meta.json")
    return json.loads(meta.read_text(encoding="utf-8")).get("commit_temp", 1.0) if meta.exists() else 1.0


def mean_steps(rows, idxs):
    return sum(rows[i]["steps"] for i in idxs) / max(len(idxs), 1)


def t_match(points, target):
    """points: [(T, steps)]. Largest T in (0, 1] with steps(T) <= target, interpolated."""
    points = sorted(points, reverse=True)
    if points[0][1] <= target:
        return points[0][0]
    for (t_hi, s_hi), (t_lo, s_lo) in zip(points, points[1:]):
        if s_lo <= target:
            return t_lo + (t_hi - t_lo) * (target - s_lo) / (s_hi - s_lo)
    return None                      # even the lowest temperature does not reach the target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    ref = read(args.ref)
    runs = [(temperature(p), p, read(p)) for p in args.runs]
    common = set(ref)
    for _, _, rows in runs:
        common &= rows.keys()
    calm = [i for i in common if all(rows[i]["gen_tokens"] < RUNAWAY for _, _, rows in runs)
            and ref[i]["gen_tokens"] < RUNAWAY]
    target = mean_steps(ref, calm)
    print(f"problems: {len(calm)}, reference steps {target:.2f}, accuracy {100 * sum(ref[i]['correct'] for i in calm) / len(calm):.1f}")
    points = []
    for t, p, rows in sorted(runs, reverse=True):
        s = mean_steps(rows, calm)
        acc = 100 * sum(rows[i]["correct"] for i in calm) / len(calm)
        points.append((t, s))
        print(f"T={t:<5} steps {s:7.2f} ({100 * (s / target - 1):+.1f}%)  accuracy {acc:.1f}  {Path(p).name}")
    tm = t_match(points, target)
    print(f"T_match = {tm:.3f}" if tm is not None else "T_match: below the lowest temperature tried")


if __name__ == "__main__":
    main()
