"""Inspect previously saved DreamReasoner ToM 8K answers, IDs 0 through 48."""
import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


PUNCT = str.maketrans({**dict.fromkeys("‐‑‒–—−", "-"), **dict.fromkeys("‘’ʼ", "'")})
STEP = re.compile(r"^\s*(?:[-*•]\s*)?step\s+(\d+)\b", re.I)


def normalize(text):
    text = unicodedata.normalize("NFKC", text).translate(PUNCT)
    return " ".join(text.casefold().split())


def parse(text):
    steps = defaultdict(list)
    for raw in text.splitlines():
        line = normalize(raw).replace("**", "").replace("`", "")
        match = STEP.match(line)
        if not match:
            continue
        body = re.search(r"\{([^{}]*)\}", line)
        fields = {}
        valid = body is not None
        if body:
            for part in body[1].split(";"):
                key, sep, value = part.partition(":")
                key, value = key.strip(), value.strip()
                if not sep or not key or not value or key in fields:
                    valid = False
                else:
                    fields[key] = value
        steps[int(match[1])].append((fields, valid, raw))
    return steps


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--longproc", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    gold_path = args.longproc / "data/tom_tracking/tom_tracking_8k.json"
    examples = json.loads(gold_path.read_text(encoding="utf-8"))
    steps, items, summary = [], [], []
    for i in range(49):
        idx = f"tom_tracking_8k:{i}"
        source = args.run / f"item_{i:04}.json"
        row = json.loads(source.read_text(encoding="utf-8"))
        if row["idx"] != idx:
            raise ValueError(f"{source}: expected {idx}")
        gold = parse(examples[i]["solution"])
        pred = parse(row["text"])
        if not gold or sorted(gold) != list(range(len(gold))):
            raise ValueError(f"{idx}: incomplete reference sequence")
        prev = None
        for s in range(len(gold)):
            [(expected, valid, gold_line)] = gold[s]
            if not valid or len(expected) != 4:
                raise ValueError(f"{idx} step {s}: invalid reference")
            [belief_key] = [key for key in expected if "'s belief on " in key]
            belief = expected[belief_key]
            candidates = pred.get(s, [])
            actual, valid, _ = candidates[0] if len(candidates) == 1 else ({}, False, "")
            status = ("missing" if not candidates else "duplicate" if len(candidates) > 1
                      else "ok" if valid and actual.keys() == expected.keys() else "format")
            group = "initial" if prev is None else "changed" if belief != prev else "unchanged"
            steps.append(dict(idx=idx, step=s, quarter=1 + 4*s//len(gold), group=group,
                              status=status, state_ok=int(status == "ok" and actual == expected),
                              belief_ok=int(valid and actual.get(belief_key) == belief),
                              gold_belief=belief, pred_belief=actual.get(belief_key, ""),
                              gold_line=gold_line, pred_lines="\n".join(c[2] for c in candidates)))
            prev = belief
        items.append(dict(idx=idx, source=str(source), reference_steps=len(gold),
                          extra_step_ids=json.dumps(sorted(pred.keys() - gold.keys())),
                          tokens=row["gen_tokens"], official_accuracy=row["metrics"]["accuracy"],
                          official_partial=row["metrics"]["partial_accuracy"]))
    for quarter in (0, 1, 2, 3, 4):
        for group in ("all", "changed", "unchanged", "initial"):
            subset = [r for r in steps if (not quarter or r["quarter"] == quarter)
                      and (group == "all" or r["group"] == group)]
            n = len(subset)
            a = sum(r["state_ok"] for r in subset)
            b = sum(r["belief_ok"] for r in subset)
            summary.append(dict(quarter=quarter or "all", group=group, steps=n,
                                questions=len({r["idx"] for r in subset}),
                                state_correct=a, belief_correct=b,
                                state_pct=100*a/n if n else "", belief_pct=100*b/n if n else "",
                                **{k: sum(r["status"] == k for r in subset)
                                   for k in ("ok", "missing", "duplicate", "format")}))
            if n and group in ("all", "changed"):
                print(f"Q{quarter or 'all'} {group:8} N={n:4}  "
                      f"state={a}/{n} ({100*a/n:.1f}%)  belief={b}/{n} ({100*b/n:.1f}%)")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("per_step", steps), ("per_question", items), ("summary", summary)):
        write_csv(args.out / f"{name}.csv", rows)
    print(f"Saved {len(items)} questions, {len(steps)} reference steps: {args.out}")


if __name__ == "__main__":
    main()
