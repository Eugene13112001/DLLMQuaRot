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
import re
import sys
from typing import Optional

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
            "gen_length": cfg.get("gen_length"),
            "reported_cut_off": payload.get("cut_off"),
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
