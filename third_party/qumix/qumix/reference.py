"""QuMix reference implementation.

A short, readable version of what the experiment code computes. It is not called by the experiments; instead,
`tests/test_reference.py` checks that every function here returns bit-identical tensors to the corresponding code in
`accuracy/sdar/qumix_eval.py` and `calibration/build_po2q.py`. Reading this file is therefore enough to follow the
method; the experiment harness adds model hooks and cache bookkeeping.

Notation (per layer): HKV KV heads, D = 128 head dimensions, keys and queries taken before RoPE.

  1. Calibration statistics  Q = sum q q^T over the queries of the heads that share a KV head,  K = sum k k^T.
  2. Query-weighted basis    Q = L L^T,  L^T K L = U diag(lambda) U^T,  sigma = sqrt(lambda / ntok).
                             Coefficients c = k W with W = L U / sigma; reconstruction k = c W_inv^T with
                             W_inv = L^{-T} U sigma. The sigma-squared-weighted error of c is then the query-weighted error of k.
  3. Bit allocation          reverse water-filling over {0,1,2,4,8,16} bits using log2(sigma) per direction.
  4. Key quantization        per 32-token group and direction: asymmetric uniform quantization of c with the
                             allocated bits (16 = exact, 0 = dropped); step and minimum stored with 8 bits.
  5. Value quantization      per token: asymmetric 2-bit quantization over the head dimension, 8-bit step/minimum.

In the cache, a group of 32 committed tokens is quantized once it is complete ("sealed"); the incomplete last group
stays exact, and the forward pass that produces keys and values uses them exactly. RoPE is applied to the
reconstructed keys.
"""
import torch

BIT_WIDTHS = (0, 1, 2, 4, 8, 16)   # bit widths a key direction can receive (16 = kept exact, 0 = dropped)
GROUP = 32                          # tokens per quantization group


# ----------------------------------------------------------------------------------------------------------------
# 1. Calibration statistics
# ----------------------------------------------------------------------------------------------------------------
def accumulate_stats(stats, q, k):
    """Add one forward pass of one layer to the calibration statistics.

    q: (HQ, T, D) pre-RoPE queries, k: (HKV, T, D) pre-RoPE keys, both float32. The HQ // HKV query heads that read
    the same KV head are pooled. `stats` is a dict with Q, K (HKV, D, D) and ntok (tokens seen), or None to start.
    """
    HKV, T, D = k.shape
    G = q.shape[0] // HKV
    qg = q.view(HKV, G, T, D).reshape(HKV, G * T, D)
    if stats is None:
        stats = dict(Q=0, K=0, ntok=0)
    stats['K'] = stats['K'] + torch.einsum('htd,hte->hde', k, k)
    stats['Q'] = stats['Q'] + torch.einsum('htd,hte->hde', qg, qg)
    stats['ntok'] += T
    return stats


# ----------------------------------------------------------------------------------------------------------------
# 2. Query-weighted basis
# ----------------------------------------------------------------------------------------------------------------
def build_basis(Q, K, ntok):
    """Return (W, W_inv, sigma2), each per KV head: W, W_inv of shape (HKV, D, D), sigma2 of shape (HKV, D).

    c = k @ W gives the coefficients; k = c @ W_inv^T reconstructs the key (W_inv = W^{-T}). Directions are sorted by
    sigma2, the spectrum of the query-weighted key second moment, in decreasing order.
    """
    D = Q.shape[-1]
    jitter = 1e-4 * Q.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None] * torch.eye(D, device=Q.device)[None]
    L = torch.linalg.cholesky(Q + jitter)                    # Q = L L^T
    L_inv = torch.linalg.inv(L)
    M = L.transpose(-1, -2) @ K @ L                          # keys as the queries see them
    M = 0.5 * (M + M.transpose(-1, -2))
    lam, U = torch.linalg.eigh(M)
    lam, U = lam.flip(-1), U.flip(-1)                        # decreasing importance
    sigma = (lam.clamp(min=0) / ntok).sqrt()
    W = (L @ U) / sigma.clamp(min=1e-8).unsqueeze(1)          # unit-second-moment coefficients
    W_inv = (L_inv.transpose(-1, -2) @ U) * sigma.unsqueeze(1)
    return W, W_inv, sigma ** 2


# ----------------------------------------------------------------------------------------------------------------
# 3. Bit allocation (reverse water-filling)
# ----------------------------------------------------------------------------------------------------------------
def _water_fill(sigma2, alphabet, target, count_scales):
    """Bits = log2(sigma) + offset, rounded to the nearest allowed width; the offset (water level) is bisected so that
    the cost (payload bits, plus one bit per kept direction when count_scales) matches the target."""
    A = torch.tensor(sorted(alphabet), dtype=torch.float, device=sigma2.device)
    mid = (A[1:] + A[:-1]) / 2
    log2_sigma = 0.5 * torch.log2(sigma2.clamp(min=1e-20))
    lo, hi = -60.0, 60.0
    for _ in range(60):
        c = (lo + hi) / 2
        b = A[torch.bucketize(log2_sigma + c, mid)]
        cost = (b.sum() + (b > 0).float().sum()) if count_scales else b.sum()
        if cost > target:
            hi = c
        else:
            lo = c
    return b.long()


def allocate_bits(sigma2, budget):
    """Bit width per direction for one layer, shape (HKV, D), for an average payload `budget` (bits per element).

    First allocate over {0,2,4,8,16} with a payload target of budget * D per head; then allocate over
    {0,1,2,4,8,16} with the same effective cost (payload + one bit of scale per kept direction), so that the 1-bit
    level does not buy extra scale budget. The paper's 1.99 / 1.69 / 1.55-bit settings use budget 1.5 / 1.0 / 0.75.
    """
    out = []
    for s2 in sigma2:                                        # one KV head at a time
        b_pow2 = _water_fill(s2, (0, 2, 4, 8, 16), budget * s2.shape[-1], count_scales=False)
        effective = b_pow2.float().sum() + (b_pow2 > 0).float().sum()
        out.append(_water_fill(s2, BIT_WIDTHS, effective.item(), count_scales=True))
    return torch.stack(out)


# ----------------------------------------------------------------------------------------------------------------
# 4. Key quantization
# ----------------------------------------------------------------------------------------------------------------
def _quantize_scale_and_min(step, minimum, scale_bits, widen, n_fix):
    """Store step sizes on a log2 grid and minima on a linear grid, each with scale_bits bits. The grid ranges are
    taken from the first n_fix groups and widened (by `widen` in log2 for steps, by widen/4 of the range for minima)."""
    Lv = 2 ** scale_bits - 1
    ls = step.log2()
    lo = ls[:, :, :n_fix].amin(2, keepdim=True) - widen
    hi = ls[:, :, :n_fix].amax(2, keepdim=True) + widen
    d = (hi - lo).clamp(min=1e-8) / Lv
    step = 2 ** ((((ls - lo) / d).round().clamp(0, Lv)) * d + lo)
    lo = minimum[:, :, :n_fix].amin(2, keepdim=True)
    hi = minimum[:, :, :n_fix].amax(2, keepdim=True)
    w = (hi - lo) * widen * 0.25
    lo = lo - w
    hi = hi + w
    d = (hi - lo).clamp(min=1e-8) / Lv
    minimum = (((minimum - lo) / d).round().clamp(0, Lv)) * d + lo
    return step, minimum


def quantize_keys(k, W, W_inv, bits, scale_bits=8, widen=1.0, fix_tokens=256):
    """Quantize and reconstruct a segment of pre-RoPE keys k: (B, HKV, T, D) with a per-direction bit map
    bits: (HKV, D). Returns the reconstructed keys (B, HKV, T, D).

    Each group of 32 tokens gets, per direction, a step and a minimum (asymmetric quantization); these are stored with
    `scale_bits` bits using ranges fixed from the first `fix_tokens` tokens of the segment. Directions with 16 bits are
    kept exact and directions with 0 bits are dropped.
    """
    c = torch.einsum('bhtd,hdr->bhtr', k, W)                # coefficients along the query-weighted directions
    bits = bits.to(c.device)
    levels = torch.where((bits > 0) & (bits < 16), 2.0 ** bits.float() - 1, torch.zeros_like(bits, dtype=torch.float))
    levels = levels[:, None, None, :]                        # (HKV, 1, 1, D)
    exact = (bits >= 16).float()[:, None, :]                 # (HKV, 1, D)
    kept = (bits > 0).float()[:, None, :]
    B, H, T, R = c.shape
    pad = (GROUP - T % GROUP) % GROUP
    cp = torch.cat([c, c[:, :, -1:].expand(B, H, pad, R)], 2) if pad else c
    cg = cp.view(B, H, -1, GROUP, R)                         # (B, HKV, groups, 32, D)
    minimum = cg.amin(3, keepdim=True)
    step = (cg.amax(3, keepdim=True) - minimum).clamp(min=1e-8) / levels.clamp(min=1)
    if scale_bits < 16:
        n_fix = max(1, min(cg.shape[2], fix_tokens // GROUP))
        step, minimum = _quantize_scale_and_min(step, minimum, scale_bits, widen, n_fix)
    q = (((cg - minimum) / step).round().clamp(torch.zeros_like(levels), levels) * step + minimum).view(B, H, -1, R)[:, :, :T]
    c_hat = torch.where(exact > 0, c, q) * (kept > 0)
    return torch.einsum('bhtr,hdr->bhtd', c_hat, W_inv)


# ----------------------------------------------------------------------------------------------------------------
# 5. Value quantization
# ----------------------------------------------------------------------------------------------------------------
def quantize_values(v, bits=2, scale_bits=8, widen=1.0, fix_tokens=256):
    """Per-token asymmetric quantization of values v: (B, HKV, T, D) over the head dimension, with the per-token step
    and minimum stored with `scale_bits` bits (ranges from the first `fix_tokens` tokens). Returns the reconstruction."""
    minimum = v.amin(-1, keepdim=True)
    step = (v.amax(-1, keepdim=True) - minimum).clamp(min=1e-8) / (2 ** bits - 1)
    n_fix = max(1, min(step.shape[2], fix_tokens))
    step, minimum = _quantize_scale_and_min(step, minimum, scale_bits, widen, n_fix)
    return ((v - minimum) / step).round().clamp(0, 2 ** bits - 1) * step + minimum


# ----------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------
def entry_to_bits(entry):
    """Convert a calibration entry {'Wqw', 'Wrw', 'groups': [(bits, mask), ...]} to (W, W_inv, bits (HKV, D))."""
    mask0 = entry['groups'][0][1]
    bits = torch.zeros(mask0.shape[0], mask0.shape[-1], dtype=torch.long, device=mask0.device)
    for b, m in entry['groups']:
        bits = bits + int(b) * m[:, 0, :].long()
    return entry['Wqw'], entry['Wrw'], bits


def kv_bits(bits, value_bits=2, tokens=4100, exact_tail=15.5):
    """Average bits per cached element (keys and values) as reported in the paper: key payload plus one 8-bit step and
    one 8-bit minimum per kept direction and 32-token group (1/2 bit per element), value payload plus 8-bit step and
    minimum per token and head (1/8 bit per element), and the exact (16-bit) tail of 15.5 tokens on average."""
    b = bits.double()
    k = (b + 0.5 * ((b > 0) & (b < 16))).mean().item()
    f = exact_tail / tokens
    return ((1 - f) * (k + value_bits + 0.125) + 32 * f) / 2
