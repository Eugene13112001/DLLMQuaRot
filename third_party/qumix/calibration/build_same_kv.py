#!/usr/bin/env python3
"""Build the "QuMix (same KV)" allocations of Table 1 (Appendix: same-KV comparison).

For each baseline, re-run QuMix's allocation so that QuMix's average KV bits (keys incl. scales, 2-bit values with
per-token scales, and the 15.5-token exact tail at a 4,100-token input) match the baseline's KV bits. The basis and
the stored spectrum of the calibration file are reused; only the bit allocation changes. Tags qmxmatch<baseline> are
added to the calibration file (same 8-bit key scales as qmx15qs8, and 8-bit value scales).

Usage: python build_same_kv.py --model sdar --src calib/sdar_calib.pt --dst calib/sdar_calib_same_kv.pt
"""
import argparse, hashlib, json
from pathlib import Path
import torch
torch.set_num_threads(2)
A = torch.tensor([0, 1, 2, 4, 8, 16], dtype=torch.float64); mids = (A[1:]+A[:-1])/2


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def allocate(logs, cost):
    lo = torch.full(logs.shape[:-1]+(1,), -100., dtype=torch.float64); hi = -lo
    for _ in range(48):
        off = (lo+hi)/2; b = A[torch.bucketize(logs+off, mids)]
        eff = (b+0.5*((b > 0) & (b < 16))).mean(-1, keepdim=True)
        mask = eff > cost; hi = torch.where(mask, off, hi); lo = torch.where(mask, lo, off)
    return A[torch.bucketize(logs+lo, mids)]


def bits(b):
    f = 15.5/4100; kb = (b+0.5*((b > 0) & (b < 16))).mean().item()
    return ((1-f)*(kb+2.125)+32*f)/2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['sdar', 'nemotron'], required=True)
    ap.add_argument('--src', required=True); ap.add_argument('--dst', required=True)
    args = ap.parse_args()
    p = Path(args.src); c = torch.load(p, map_location='cpu', weights_only=False); P = c['PROJ']; layers = sorted(P)
    logs = torch.stack([0.5*torch.log2(P[l]['sig2'].double().clamp(min=1e-20)) for l in layers])
    rows = []
    targets = [('svdq10', 1.88), ('kivi1', 2.15 if args.model == 'sdar' else 2.18), ('int2', 2.30), ('svdq15', 2.38),
               ('kivi2', 2.94), ('mixkvq', 3.28), ('fp8', 8.15)]
    for name, target in targets:
        low, high = 0., 16.; best = None
        for _ in range(32):
            mid = (low+high)/2; b = allocate(logs, mid)
            if bits(b) > target: high = mid
            else: low = mid; best = b
        actual = bits(best); assert 0 <= target-actual < 0.03, (args.model, name, actual)
        tag = 'qmxmatch'+name
        for j, l in enumerate(layers):
            entry = dict(P[l]['qmx15qs8']); entry['groups'] = [(int(v), (best[j] == v).float().unsqueeze(1)) for v in A.tolist() if v > 0 and (best[j] == v).any()]
            entry['vsbits'] = 8; P[l][tag] = entry
        hist = {str(int(v)): int((best == v).sum()) for v in A}
        rows.append(dict(baseline=name, target_bits=target, actual_bits=actual, tag=tag, histogram=hist, vbits=2))
    dst = Path(args.dst); torch.save(c, dst)
    print(json.dumps(dict(source_sha256=sha(p), calib_sha256=sha(dst), rows=rows), indent=1))
    print(args.model, [(r['baseline'], round(r['actual_bits'], 6)) for r in rows], flush=True)


if __name__ == '__main__':
    main()
