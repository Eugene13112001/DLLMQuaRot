"""Re-score saved completions with the current answer extractor.

Generation is the expensive part of a GSM8K run; parsing the answer out of the
completion is free. Keeping the two separable means a fix to extraction costs
seconds instead of hours, and it makes the size of that fix visible -- which
matters, because a parser bug otherwise gets silently attributed to the
quantization method.

    python scripts/rescore.py out/gsm8k_quarot.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dllmquant.eval import extract_answer  # noqa: E402
from dllmquant.eval.gsm8k import EvalResult  # noqa: E402
from dllmquant.report import round_floats, sibling_csv, write_csv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="JSON files written by evaluate.py")
    ap.add_argument("--write", action="store_true",
                    help="update the files in place with the new scoring")
    args = ap.parse_args()

    for path in args.results:
        p = pathlib.Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        samples = data.get("samples", [])
        if not samples:
            print(f"{p.name}: no samples stored, nothing to re-score")
            continue
        if not any("completion" in s for s in samples):
            print(f"{p.name}: samples have no completions "
                  "(written before they were saved in full)")
            continue

        changed, correct = [], 0
        for s in samples:
            pred = extract_answer(s.get("completion", ""))
            gold = s.get("gold")
            hit = pred is not None and gold is not None and abs(pred - gold) < 1e-4
            if pred != s.get("pred") or hit != s.get("correct"):
                changed.append({
                    "gold": gold, "was_pred": s.get("pred"), "now_pred": pred,
                    "was_correct": s.get("correct"), "now_correct": hit,
                })
            s["pred"], s["correct"] = pred, hit
            correct += int(hit)

        n = len(samples)
        old = data.get("accuracy", float("nan"))
        print(f"\n{p.name}: {data.get('correct', '?')}/{n} = {100 * old:.2f}%"
              f"  ->  {correct}/{n} = {100 * correct / n:.2f}%")

        # The header and the stored answers can disagree -- a run cut short
        # mid-write, or two runs merged. The old print showed both numbers side
        # by side and left it to the reader to notice that 87 correct out of 20
        # is impossible. Say it instead: a file in this state cannot support
        # either figure, and quietly re-scoring the surviving 20 answers
        # manufactures a third one that looks just as citable.
        stated_total = data.get("total")
        stated_correct = data.get("correct")
        problems = []
        if isinstance(stated_total, int) and stated_total != n:
            problems.append(f"header says {stated_total} answers, {n} are stored")
        if isinstance(stated_correct, int) and stated_correct > n:
            problems.append(f"header claims {stated_correct} correct out of {n}")
        if problems:
            print("  !! INCONSISTENT: " + "; ".join(problems))
            print("     Neither the header nor the re-scored figure describes a "
                  "complete run. Find out what produced this file before using "
                  "any number from it.")
        gained = sum(1 for c in changed if c["now_correct"] and not c["was_correct"])
        lost = sum(1 for c in changed if c["was_correct"] and not c["now_correct"])
        print(f"  {len(changed)} parses changed: +{gained} correct, -{lost}")

        # Reported here too, because the completions of every past run are
        # stored: a result recorded before this metric existed can still be
        # asked how much of its error was the generation budget, without
        # re-running a single forward pass.
        result = EvalResult(correct=correct, total=n, samples=samples)
        if result.cut_off:
            print(f"  {result.cut_off} replies cut off mid-answer; they are "
                  f"{result.cut_off_wrong} of the {n - correct} errors "
                  f"({100 * result.cut_off_wrong / max(n - correct, 1):.0f}%)")
        budget = data.get("config", {}).get("gen_length")
        print(f"  generation budget: "
              f"{budget if budget is not None else 'not recorded in this file'}")

        for c in changed[:10]:
            print(f"    gold {c['gold']}: {c['was_pred']} -> {c['now_pred']}"
                  f"  ({c['was_correct']} -> {c['now_correct']})")

        if args.write:
            data["accuracy"] = correct / max(n, 1)
            data["correct"] = correct
            data["total"] = n
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
            csv_path = write_csv(
                sibling_csv(p),
                [
                    round_floats({
                        "question": s["question"].replace("\n", " ")[:200],
                        "predicted": s["pred"], "gold": s["gold"],
                        "correct": s["correct"],
                    })
                    for s in samples
                ],
            )
            print(f"  updated {p} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
