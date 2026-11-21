#!/usr/bin/env python3
"""Step 3 of calibration: add the 8-bit-scale variants used in the paper.

For each layer and each tag qmx15q / qmx10q / qmx075q, add qmx15qs8 / qmx10qs8 / qmx075qs8: the same basis and bit
allocation, with the per-(token group, direction) key scales and minima quantized to 8 bits on a log grid whose range is
fixed from the first 256 tokens and widened by one step (sbits=8, widen=1.0, fixT=256). With --vsbits, the tags also
record the value-scale bit width (the Nemotron harness reads it from the tag; the SDAR harness takes --vsbits 8).
"""
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('dst')
    ap.add_argument('--tags', default='qmx15q,qmx10q,qmx075q')
    ap.add_argument('--vsbits', type=int, default=0)
    args = ap.parse_args()
    C = torch.load(args.src, map_location='cpu', weights_only=False)
    for l in sorted(C['PROJ']):
        for t in args.tags.split(','):
            P = C['PROJ'][l][t]
            d = dict(Wqw=P['Wqw'], Wrw=P['Wrw'], groups=P['groups'], sbits=8, widen=1.0, fixT=256)
            if args.vsbits:
                d['vsbits'] = args.vsbits
            C['PROJ'][l][t + 's8'] = d
    torch.save(C, args.dst)
    print(f"added {[t + 's8' for t in args.tags.split(',')]} to {len(C['PROJ'])} layers -> {args.dst}")


if __name__ == '__main__':
    main()
