# -*- coding: utf-8 -*-
"""Where in the rotary spectrum does the key bias sit, and how much of it survives
a mean over the positions of one cache write?

The objection this answers: if the large |beta| pairs are the low-frequency ones, the
rotated bias is locally constant over a write, a zero point (or a window mean) absorbs it,
and the positional reading of our 2-bit gap has no numerical content.

Reads only the k_proj.bias tensors, by HTTP range request. No forward pass, no download.
"""
import json, struct, urllib.request
import numpy as np

REPO = "Efficient-Large-Model/Fast_dLLM_v2_7B"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
HEAD_DIM = 128
THETA = 1.0e6


def get(url, start=None, end=None):
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    return urllib.request.urlopen(req, timeout=60).read()


def shard_header(fn):
    n = struct.unpack("<Q", get(BASE + fn, 0, 7))[0]
    return json.loads(get(BASE + fn, 8, 8 + n - 1)), 8 + n


def bf16(raw, count):
    u = np.frombuffer(raw, dtype="<u2", count=count).astype(np.uint32) << 16
    return u.view(np.float32) if u.dtype == np.uint32 else u.astype(np.uint32).view(np.float32)


idx = json.loads(get(BASE + "model.safetensors.index.json"))["weight_map"]
names = sorted((k for k in idx if k.endswith("k_proj.bias")),
               key=lambda s: int(s.split(".")[2]))
print(f"{len(names)} bias tensors\n")

hdrs = {}
rows = []
for name in names:
    fn = idx[name]
    if fn not in hdrs:
        hdrs[fn] = shard_header(fn)
    hdr, off = hdrs[fn]
    meta = hdr[name]
    s, e = meta["data_offsets"]
    raw = get(BASE + fn, off + s, off + e - 1)
    n = int(np.prod(meta["shape"]))
    b = bf16(raw, n) if meta["dtype"] == "BF16" else np.frombuffer(raw, dtype="<f4", count=n)
    rows.append((int(name.split(".")[2]), b.astype(np.float64)))

# rotary frequency of pair c (HF sliced layout: partners are c and c + head_dim/2)
half = HEAD_DIM // 2
theta = THETA ** (-np.arange(half) / half)          # theta_c, c = 0..63


def residual_fraction(amp, P):
    """Of the bias energy, how much survives subtracting the mean over P positions?

    The stored term for a pair is amp*cos(theta*m + phi). Subtracting the window mean
    leaves amp*(cos(theta*m+phi) - mean). Averaged over the window phase, report the RMS
    of what is left as a fraction of amp -- that is what still inflates the group's range
    and what an exact per-position removal recovers.
    """
    m = np.arange(P)
    out = np.zeros_like(amp)
    for i, th in enumerate(theta):
        # average over 24 starting phases so the answer does not depend on where the
        # window happens to land in the cycle
        r = []
        for phi in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            v = np.cos(th * m + phi)
            r.append(np.sqrt(((v - v.mean()) ** 2).mean()))
        out[i] = np.mean(r)
    return out * amp


print(f"{'layer':>5} {'max|b|':>8} {'argmax pair':>11} {'theta at argmax':>15} "
      f"{'resid/P=32':>11} {'resid/P=128':>12} {'as % of L2':>11}")
tot = []
for li, b in rows:
    bh = b.reshape(-1, HEAD_DIM)                     # [kv_heads, head_dim]
    amp = np.sqrt(bh[:, :half] ** 2 + bh[:, half:] ** 2)   # [kv_heads, 64]
    h, c = np.unravel_index(np.argmax(amp), amp.shape)
    r32 = np.stack([residual_fraction(amp[i], 32) for i in range(amp.shape[0])])
    r128 = np.stack([residual_fraction(amp[i], 128) for i in range(amp.shape[0])])
    l2 = np.sqrt((amp ** 2).sum())
    pct = 100 * np.sqrt((r128 ** 2).sum()) / max(l2, 1e-12)
    tot.append((li, float(np.abs(b).max()), int(c), float(theta[c]),
                float(np.sqrt((r32 ** 2).sum())), float(np.sqrt((r128 ** 2).sum())),
                float(l2), pct))
    print(f"{li:>5} {np.abs(b).max():>8.1f} {c:>11} {theta[c]:>15.2e} "
          f"{np.sqrt((r32**2).sum()):>11.2f} {np.sqrt((r128**2).sum()):>12.2f} {pct:>10.1f}%")

print("\n--- amplitude-weighted rotary spectrum, all layers ---")
A = np.zeros(half)
for _, b in rows:
    bh = b.reshape(-1, HEAD_DIM)
    A += np.sqrt(bh[:, :half] ** 2 + bh[:, half:] ** 2).sum(axis=0)
A /= A.sum()
for lo, hi, lab in ((0, 16, "pairs 0-15  (high freq)"), (16, 32, "pairs 16-31"),
                    (32, 48, "pairs 32-47"), (48, 64, "pairs 48-63 (low freq)")):
    print(f"  {lab:28} {100*A[lo:hi].sum():5.1f}% of bias amplitude, "
          f"theta {theta[lo]:.1e}..{theta[hi-1]:.1e}")

print("\n--- how much of the bias a window mean cannot remove ---")
for P in (8, 32, 64, 128, 512):
    num = den = 0.0
    for _, b in rows:
        bh = b.reshape(-1, HEAD_DIM)
        amp = np.sqrt(bh[:, :half] ** 2 + bh[:, half:] ** 2)
        r = np.stack([residual_fraction(amp[i], P) for i in range(amp.shape[0])])
        num += (r ** 2).sum(); den += (amp ** 2).sum()
    print(f"  write of {P:>4} positions: {100*np.sqrt(num/den):5.1f}% of bias amplitude survives")
