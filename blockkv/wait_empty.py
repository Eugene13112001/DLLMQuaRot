"""Wait for an empty card and print its number (our CUDA_VISIBLE_DEVICES index) on stdout.

    CARD=$(python -m blockkv.wait_empty --cards 0 1 2 3 4 --need-gb 120)

Empty = nvidia-smi utilization <= --max-util and at least --need-gb free, --checks checks in
a row, one check every --sleep seconds (same rule as blockkv.bitsieve_timing). Progress goes
to stderr: changes only, plus a summary every 10 minutes.
"""
import argparse
import contextlib
import sys

from blockkv.bitsieve_timing import card_uuid, wait_idle


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--max-util", type=int, default=5)
    ap.add_argument("--need-gb", type=float, default=120)
    ap.add_argument("--checks", type=int, default=3)
    ap.add_argument("--sleep", type=int, default=60)
    args = ap.parse_args(argv)
    with contextlib.redirect_stdout(sys.stderr):
        uuids = {c: card_uuid(c) for c in args.cards}
        card = wait_idle(args.cards, uuids, args.max_util, args.need_gb, args.checks, args.sleep)
    print(card)


if __name__ == "__main__":
    main()
