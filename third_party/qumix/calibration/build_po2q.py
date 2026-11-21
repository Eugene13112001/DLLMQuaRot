#!/usr/bin/env python3
"""Step 2 of calibration: re-allocate key bits over the power-of-two set {0,1,2,4,8,16}.

Input: a calibration file written by `accuracy/sdar/qumix_eval.py --calibonly --ladder --savecalib` (or the Nemotron
builder), which stores per layer the query-weighted basis (tag 'qmx075': Wqw, Wrw) and its spectrum 'sig2'.
No GPU and no recapture are needed.

For each budget bbar (average payload bits per key element) and each KV head:
  * 'p' tags: reverse water-filling over {0,2,4,8,16} with a payload target of bbar bits per direction;
  * 'q' tags: the same rule over {0,1,2,4,8,16}, with the bisection target set to the EFFECTIVE bits of the 'p'
    allocation (payload + 1 bit of scale per kept direction), so that the 1-bit level cannot buy extra scale budget.
The paper's QuMix settings use the 'q' tags qmx15q / qmx10q / qmx075q (1.99 / 1.69 / 1.55 average KV bits after
adding values, scales and the exact tail; see add_scale_tags.py for the 8-bit scale variants).
"""
import argparse
import torch

ALPHA_P = (0, 2, 4, 8, 16)
ALPHA_Q = (0, 1, 2, 4, 8, 16)
BUDGETS = ((1.5, '15'), (1.0, '10'), (0.75, '075'), (0.5, '05'), (0.375, '0375'))


def alloc_payload(sig2, bbar, alpha):            # payload-target bisection (same rule as alloc_bits in qumix_eval.py)
    A = torch.tensor(sorted(alpha), dtype=torch.float, device=sig2.device)
    mid = (A[1:]+A[:-1])/2
    log2s = 0.5*torch.log2(sig2.clamp(min=1e-20)); lo, hi = -60.0, 60.0
    tgt = bbar*sig2.shape[-1]
    for _ in range(60):
        c = (lo+hi)/2; b = A[torch.bucketize(log2s+c, mid)]
        if b.sum() > tgt: hi = c
        else: lo = c
    return b.long()


def alloc_effective(sig2, eff_tgt, alpha):       # bisect on payload + 1.0*kept (key scale cost)
    A = torch.tensor(sorted(alpha), dtype=torch.float, device=sig2.device)
    mid = (A[1:]+A[:-1])/2
    log2s = 0.5*torch.log2(sig2.clamp(min=1e-20)); lo, hi = -60.0, 60.0
    for _ in range(60):
        c = (lo+hi)/2; b = A[torch.bucketize(log2s+c, mid)]
        if (b.sum() + (b > 0).float().sum()) > eff_tgt: hi = c
        else: lo = c
    return b.long()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('dst')
    args = ap.parse_args()
    C = torch.load(args.src, map_location='cpu', weights_only=False)
    P = C['PROJ']
    tags = [(bb, f'qmx{s}p', f'qmx{s}q') for bb, s in BUDGETS]
    stats = {}
    for i in sorted(P):
        sig2 = P[i]['sig2']                          # (HKV, hd)
        Wqw, Wrw = P[i]['qmx075']['Wqw'], P[i]['qmx075']['Wrw']
        HKV = sig2.shape[0]
        for bbar, ptag, qtag in tags:
            BP = torch.stack([alloc_payload(sig2[h], bbar, ALPHA_P) for h in range(HKV)])
            eff = BP.float().sum(-1) + (BP > 0).float().sum(-1)      # per-head effective target
            BQ = torch.stack([alloc_effective(sig2[h], eff[h].item(), ALPHA_Q) for h in range(HKV)])
            for tag, BA in ((ptag, BP), (qtag, BQ)):
                ga = [(int(b), (BA == b).float().unsqueeze(1)) for b in BA.unique().tolist() if b > 0]
                P[i][tag] = dict(Wqw=Wqw, Wrw=Wrw, groups=ga)
                st = stats.setdefault(tag, [])
                st.append((BA.float().mean().item(), (BA > 0).float().mean().item(), (BA == 1).float().mean().item()))
    torch.save(C, args.dst)
    T, TAIL = 4100.0, 15.5                           # accounting at a 4K input with the exact tail (as in the paper)
    for bbar, ptag, qtag in tags:
        for tag in (ptag, qtag):
            st = stats[tag]
            pay = sum(s[0] for s in st)/len(st); kept = sum(s[1] for s in st)/len(st); one = sum(s[2] for s in st)/len(st)
            k_bare = pay + 1.0*kept
            k = k_bare + (TAIL/T)*(16-k_bare)
            v = 2.25 + (TAIL/T)*(16-2.25)
            print(f"{tag:9s}: payload {pay:.3f}  kept {kept:.3f}  1bit-frac {one:.3f}  ->  K {k:.3f}  KV {(k+v)/2:.3f}")
    print(f"saved {args.dst}")


if __name__ == '__main__':
    main()
