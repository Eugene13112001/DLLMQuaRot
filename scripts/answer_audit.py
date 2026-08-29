"""Split a GSM8K eval's failures into reasoning errors and bookkeeping.

The headline accuracy of `evaluate.py` mixes three different things:

    the model reasoned wrong
    the model reasoned right and never emitted the answer marker
    the model ran out of canvas mid-deliberation

Only the first is a property of the configuration under test. The other two
grow with degradation -- a damaged model rambles, loops, and stops before it
reaches "The answer is N" -- so a comparison of bit widths on raw accuracy
charges quantization twice for the same failure and overstates its cost.

`cut_off` in the eval does not catch this. Its regex fires only on a reply
ending in an operator or a dangling function word, and its own comment calls
itself a floor. Real truncations here end mid-number ("Adrien earned $4"),
mid-word, or on an ordinary noun, and none of those are counted.

Reads the JSON `evaluate.py --out` already wrote. No GPU, no model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from typing import Dict, Optional

# The instruction given to the model asks for a line "The answer is N". A
# completion without one did not reach its own conclusion, whatever else it
# did -- which is a far better truncation signal than the operator regex.
_MARKER = re.compile(r"(?:final\s+answer|answer)\s*(?:is|:)", re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\{")

# A finished reply ends on sentence punctuation or a closing delimiter --
# but the instructed format is "The answer is N", which ends on a digit, so
# this test alone flags almost every *correct* reply. It is only meaningful
# on replies that never reached the marker: there, ending mid-number or
# mid-noun is the signature of running out of canvas.
_ENDS_CLEAN = re.compile(r"[.!?)\]\}\"'*\s]$")

_NUMBER = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def to_float(raw: str) -> Optional[float]:
    try:
        return float(raw.replace("$", "").replace(",", ""))
    except ValueError:
        return None


def mentions_gold(text: str, gold: float, tail_chars: int) -> bool:
    """Does the gold value appear near the end of the reply?

    Deliberately weak evidence: it says the right number was on the page
    where a conclusion belongs, not that the model concluded it. Used only
    to size the extraction problem, never to re-score anything.
    """
    tail = text[-tail_chars:]
    for raw in _NUMBER.findall(tail):
        value = to_float(raw)
        if value is not None and abs(value - gold) < 1e-6:
            return True
    return False


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value on the discordant pairs.

    Only questions the two runs disagree about carry information: the ones
    both got right and both got wrong say nothing about which is better.
    Under the null the b + c disagreements split like a fair coin.

    This is the test to use here rather than comparing two accuracies. The
    same 200 questions are asked of every configuration, so the comparison
    is paired, and pairing is worth a great deal at this sample size: two
    unpaired accuracies near 90% carry a standard error of about 2.1 points
    each, which would hide exactly the one-to-two-point effects the weight
    sweep is looking for.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def by_question(samples) -> Dict[str, dict]:
    return {s.get("question", ""): s for s in samples}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dumps", nargs="+",
                    help="one or more JSON files written by evaluate.py --out")
    ap.add_argument("--tail-chars", type=int, default=200,
                    help="how much of the reply's end counts as 'near the "
                         "conclusion' when looking for the gold value")
    args = ap.parse_args()

    rows = []
    for path in args.dumps:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)

        cfg = payload.get("config", {})
        samples = payload.get("samples", [])
        if not samples:
            print(f"{path}: no samples in the dump")
            continue

        label = (f"W{cfg.get('w_bits', '?')}A{cfg.get('a_bits', '?')}"
                 f" g{cfg.get('a_group_size', '?')}")

        total = len(samples)
        correct = sum(1 for s in samples if s.get("correct"))
        no_marker = 0
        no_marker_wrong = 0
        unfinished = 0
        unfinished_wrong = 0
        recoverable = 0

        for s in samples:
            text = s.get("completion", "")
            wrong = not s.get("correct")

            marked = bool(_MARKER.search(text) or _BOXED.search(text))
            if not marked:
                no_marker += 1
                no_marker_wrong += wrong

            # Truncation, not merely an unmarked reply: it never reached the
            # instructed closing line AND it stops mid-sentence.
            if not marked and not _ENDS_CLEAN.search(text):
                unfinished += 1
                unfinished_wrong += wrong

            # Wrong, no marker, and the gold value sitting near the end: the
            # signature of a reply the extractor mis-read rather than a reply
            # the model got wrong.
            if wrong and not marked:
                gold = s.get("gold")
                if gold is not None and mentions_gold(text, float(gold),
                                                      args.tail_chars):
                    recoverable += 1

        wrong_total = total - correct
        rows.append({
            "path": path, "label": label, "total": total,
            "acc": 100.0 * correct / total,
            "wrong": wrong_total,
            "no_marker": no_marker, "no_marker_wrong": no_marker_wrong,
            "unfinished": unfinished, "unfinished_wrong": unfinished_wrong,
            "recoverable": recoverable,
            "acc_ceiling": 100.0 * (correct + recoverable) / total,
            # accuracy = (share that concluded) x (accuracy among those).
            # The two factors are different failure modes: a configuration
            # can lose points by reasoning wrong or by never getting to an
            # answer, and only the first is what the metric is meant to
            # report. Publish the pair, never either half alone --
            # conditioning on "concluded" conditions on a consequence of the
            # treatment, so the right-hand factor is descriptive, not causal.
            "concluded": 100.0 * (total - no_marker) / total,
            "acc_given": (100.0 * correct / (total - no_marker)
                          if total - no_marker else float("nan")),
            "gen_length": cfg.get("gen_length"),
            "reported_cut_off": payload.get("cut_off"),
            "samples": samples,
        })

    if not rows:
        return 1

    print()
    print("=== what the accuracy is made of ===")
    print(f"{'config':>14} {'n':>5} {'acc%':>7} {'wrong':>6} "
          f"{'no mark':>8} {'unfin':>6} {'recov':>6} {'ceiling%':>9}")
    print("-" * 70)
    for r in rows:
        print(f"{r['label']:>14} {r['total']:>5} {r['acc']:>7.2f} "
              f"{r['wrong']:>6} {r['no_marker']:>8} {r['unfinished']:>6} "
              f"{r['recoverable']:>6} {r['acc_ceiling']:>9.2f}")

    print()
    print("=== accuracy split into its two factors ===")
    print(f"{'config':>14} {'concluded%':>11} {'acc|concluded%':>16} "
          f"{'acc%':>7}")
    print("-" * 52)
    for r in rows:
        print(f"{r['label']:>14} {r['concluded']:>11.2f} "
              f"{r['acc_given']:>16.2f} {r['acc']:>7.2f}")
    print()
    print("  A configuration can lose points two ways: by reasoning wrong, "
          "or by never")
    print("  arriving at an answer. Only the first is what the metric is "
          "meant to report.")
    print("  Read the pair together and never either half alone -- "
          "'concluded' is itself")
    print("  affected by the configuration, so the right-hand column is "
          "descriptive, not")
    print("  a causal effect: if a damaged run only finishes the easy "
          "questions, its")
    print("  conditional accuracy is flattered.")

    print()
    print("  no mark  replies with no 'answer is' / \\boxed line. The answer "
          "was scored")
    print("           off the last number in the text, which is a guess.")
    print("  unfin    subset of 'no mark' that also stops mid-sentence: "
          "ran out of canvas.")
    print("           'no mark' minus 'unfin' ended cleanly without the "
          "marker -- those are")
    print("           extractor failures, not truncations.")
    print("  recov    wrong AND unmarked AND the gold value sits in the last "
          "few lines:")
    print("           the extractor's failure, not the model's. A floor, not "
          "a count.")
    print("  ceiling  accuracy if every 'recov' reply were scored correct.")

    for r in rows:
        if r["reported_cut_off"] == 0 and r["unfinished"] > 0:
            print()
            print(f"  {r['path']}: the eval reported cut_off = 0, but "
                  f"{r['unfinished']} replies do not end on sentence "
                  f"punctuation.")
            print("  The eval's cut-off regex only fires on a trailing "
                  "operator or function word;")
            print("  truncations that stop mid-number or mid-noun are "
                  "invisible to it. Raising")
            print("  gen_length on the strength of cut_off = 0 is not "
                  "supported by that number.")

    if len(rows) > 1:
        base = rows[0]
        print()
        print("=== damage against the first dump, raw and corrected ===")
        print(f"{'config':>14} {'raw':>8} {'corrected':>11}")
        print("-" * 36)
        for r in rows[1:]:
            print(f"{r['label']:>14} "
                  f"{base['acc'] - r['acc']:>8.2f} "
                  f"{base['acc_ceiling'] - r['acc_ceiling']:>11.2f}")
        print()
        print("  If the two columns disagree, part of the measured cost of "
              "quantization is")
        print("  the extractor, not the model, and the raw column must not "
              "go in the chart.")

        print()
        print("=== paired test against the first dump (McNemar, exact) ===")
        print(f"{'config':>14} {'paired':>7} {'lost':>5} {'gained':>7} "
              f"{'p':>8}")
        print("-" * 46)
        base_by_q = by_question(base["samples"])
        for r in rows[1:]:
            other = by_question(r["samples"])
            shared = [q for q in base_by_q if q in other]
            lost = sum(1 for q in shared
                       if base_by_q[q].get("correct")
                       and not other[q].get("correct"))
            gained = sum(1 for q in shared
                         if not base_by_q[q].get("correct")
                         and other[q].get("correct"))
            p = mcnemar(lost, gained)
            print(f"{r['label']:>14} {len(shared):>7} {lost:>5} "
                  f"{gained:>7} {p:>8.3f}")
        print()
        print("  lost/gained  questions the first dump got right and this "
              "one wrong, and back.")
        print("  Only those carry information. A p above 0.05 means this "
              "configuration is")
        print("  not measurably worse -- which for four-bit weights is the "
              "expected result,")
        print("  and is a claim the raw accuracy column cannot make on its "
              "own.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
