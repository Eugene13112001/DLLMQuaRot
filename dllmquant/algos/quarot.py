"""QuaRot-style orthogonal rotations, adapted to masked diffusion LMs.

Reference: "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs"
(arXiv:2404.00456, Ashkboos et al.).

The idea is computational invariance.  For an orthogonal ``Q``, inserting
``Q Qᵀ = I`` between the residual stream and every layer that reads or writes
it leaves the network's function unchanged, but the activations themselves are
now a *rotated* basis in which no single channel carries an outlier: a
Hadamard rotation spreads one large coordinate across all of them.  Per-token
activation quantization, which is defeated by one fat channel, then works.

This complements the paper's IA-AQ rather than duplicating it:

    IA-AQ   acts along the TOKEN axis   -- whose dynamic range to preserve
    QuaRot  acts along the CHANNEL axis -- so no channel needs preserving

What is diffusion-specific here
-------------------------------
The algebra of computational invariance does not care whether attention is
causal, so the rotations transfer to LLaDA unchanged.  Three things do change:

1. **The mask embedding gets rotated with everything else.**  At early
   denoising steps most positions carry the *same* vector -- the ``[MASK]``
   embedding -- so the activation distribution is dominated by one row of the
   embedding matrix.  That row is rotated too (it is a row of ``E``), which is
   what keeps invariance exact at high mask ratios; ``validate_rotation``
   checks it explicitly instead of assuming it.

2. **Invariance must be verified across the trajectory, not at one state.**
   A dense LLM has one activation regime; a DLLM sweeps from all-[MASK] to
   fully decoded.  ``validate_rotation`` therefore checks equivalence and
   measures outlier reduction at several mask ratios.

3. **Ordering with the other three components.**  Rotation must be applied
   *before* TMAS calibration, so that CGQ's Hessians and IA-AQ's value scales
   are both computed in the rotated basis.  Rotating afterwards would silently
   invalidate every calibrated scale.

Only RMSNorm is supported, as in the original: LayerNorm's mean subtraction is
a projection that does not commute with an arbitrary rotation.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------- Hadamard


def _sylvester(n: int, device=None, dtype=torch.float64) -> torch.Tensor:
    """Unnormalised Hadamard of order n, n a power of two."""
    h = torch.ones(1, 1, device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return h


def _legendre(a: int, q: int) -> int:
    if a % q == 0:
        return 0
    return 1 if pow(a, (q - 1) // 2, q) == 1 else -1


def _paley(q: int, device=None, dtype=torch.float64) -> torch.Tensor:
    """Unnormalised Hadamard of order q+1 for prime q = 3 (mod 4)."""
    if q % 4 != 3:
        raise ValueError(f"Paley type I needs q = 3 (mod 4), got {q}")
    n = q + 1
    jac = torch.tensor(
        [[_legendre(i - j, q) for j in range(q)] for i in range(q)],
        device=device,
        dtype=dtype,
    )
    s = torch.zeros(n, n, device=device, dtype=dtype)
    s[0, 1:] = 1
    s[1:, 0] = -1
    s[1:, 1:] = jac
    return torch.eye(n, device=device, dtype=dtype) + s


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# Paley-constructible orders that show up as MLP widths in practice
# (12288 = 12 x 1024, 5120 = 20 x 256, ...).
_PALEY_ORDERS = {12: 11, 20: 19, 24: 23, 28: 27, 32: 31, 44: 43, 48: 47, 60: 59}


def hadamard_factors(n: int) -> Optional[Tuple[int, int]]:
    """Split n into (m, 2^k) with an m we can build a Hadamard for."""
    if _is_pow2(n):
        return (1, n)
    for m in sorted(_PALEY_ORDERS):
        if n % m == 0 and _is_pow2(n // m):
            q = _PALEY_ORDERS[m]
            if m == q + 1 and q % 4 == 3 and _is_prime(q):
                return (m, n // m)
    return None


def _is_prime(x: int) -> bool:
    if x < 2:
        return False
    for i in range(2, int(x**0.5) + 1):
        if x % i == 0:
            return False
    return True


def hadamard_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Orthonormal Hadamard of order n, or a random orthogonal fallback."""
    factors = hadamard_factors(n)
    if factors is None:
        warnings.warn(
            f"no Hadamard construction for n={n}; falling back to a random "
            "orthogonal matrix (still exactly invariant, but it spreads "
            "outliers less evenly than a Hadamard)",
            RuntimeWarning,
        )
        a = torch.randn(n, n, device=device, dtype=torch.float64)
        q, r = torch.linalg.qr(a)
        q = q * torch.sign(torch.diag(r)).unsqueeze(0)
        return q.to(dtype)

    m, p = factors
    h = _sylvester(p, device=device)
    if m > 1:
        h = torch.kron(_paley(_PALEY_ORDERS[m], device=device), h)
    return (h / math.sqrt(n)).to(dtype)


def random_hadamard_matrix(n: int, device=None, dtype=torch.float32, seed: int = 0):
    """Hadamard with random sign flips -- QuaRot's randomised variant."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(dtype)
    h = hadamard_matrix(n, device=device, dtype=dtype)
    return h * signs.to(h.device).unsqueeze(0)


def fast_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal Hadamard to the last axis without materialising it.

    Needed for the online rotation in front of the MLP down-projection, where
    the width (12288 for LLaDA-8B) makes an explicit matrix a 600 MB object.
    """
    n = x.shape[-1]
    factors = hadamard_factors(n)
    if factors is None:
        raise ValueError(f"no Hadamard construction for n={n}")
    m, p = factors

    orig = x.shape
    y = x.reshape(-1, m, p) if m > 1 else x.reshape(-1, 1, p)

    # Butterfly over the power-of-two axis: O(p log p), no matrix.
    step = 1
    while step < p:
        y = y.reshape(y.shape[0], y.shape[1], -1, 2, step)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack([a + b, a - b], dim=-2)
        y = y.reshape(y.shape[0], y.shape[1], -1)
        step *= 2

    if m > 1:
        hm = _paley(_PALEY_ORDERS[m], device=x.device).to(x.dtype)
        # y @ H_m along the m axis, NOT H_m @ y: Paley matrices are not
        # symmetric, unlike Sylvester ones, so the side matters and a
        # power-of-two-only test would never catch it.
        y = torch.einsum("ji,bjp->bip", hm, y)

    return (y.reshape(orig) / math.sqrt(n)).to(x.dtype)


# ------------------------------------------------------------ norm fusion


def fuse_norm_into_linears(norm: nn.Module, consumers: Sequence[nn.Linear]) -> None:
    """Fold an RMSNorm's elementwise gain into the layers that read it.

    ``RMSNorm(h) * g`` followed by ``W`` equals bare ``RMSNorm(h)`` followed by
    ``W * g``.  Only after this does the norm commute with a rotation, because
    plain RMS normalisation depends on ‖h‖, which a rotation preserves, while
    the per-channel gain does not.
    """
    weight = getattr(norm, "weight", None)
    if weight is None:
        return  # already bare

    if hasattr(norm, "bias") and getattr(norm, "bias") is not None:
        raise NotImplementedError(
            "norm has a bias term; QuaRot's invariance argument assumes "
            "RMSNorm without bias"
        )

    g = weight.data.to(torch.float64)
    for lin in consumers:
        if lin.in_features != g.numel():
            raise ValueError(
                f"norm width {g.numel()} does not match consumer in_features "
                f"{lin.in_features}"
            )
        lin.weight.data = (
            lin.weight.data.to(torch.float64) * g.unsqueeze(0)
        ).to(lin.weight.dtype)

    weight.data = torch.ones_like(weight.data)


# ------------------------------------------------------------- rotations


@dataclass
class RotationPlan:
    """Which modules participate in the residual-stream rotation."""

    embeddings: List[nn.Module]
    input_linears: List[nn.Linear]      # read the residual: W <- W @ Q
    output_linears: List[nn.Linear]     # write the residual: W <- Qᵀ @ W
    norm_groups: List[Tuple[nn.Module, List[nn.Linear]]]
    # (fused-or-v projection, out projection, v column offset or None)
    head_pairs: List[Tuple[nn.Linear, nn.Linear, Optional[int]]]

    def check(self, d_model: int) -> None:
        for e in self.embeddings:
            if e.weight.shape[-1] != d_model:
                raise ValueError(
                    f"embedding width {e.weight.shape[-1]} != d_model {d_model}"
                )
        for lin in self.input_linears:
            if lin.in_features != d_model:
                raise ValueError(
                    f"input linear expects {lin.in_features}, d_model is {d_model}"
                )
        for lin in self.output_linears:
            if lin.out_features != d_model:
                raise ValueError(
                    f"output linear yields {lin.out_features}, d_model is {d_model}"
                )


@torch.no_grad()
def rotate_residual_stream(plan: RotationPlan, q: torch.Tensor) -> None:
    """Apply h -> h @ Q throughout, leaving the function unchanged."""
    d = q.shape[0]
    plan.check(d)
    qd = q.to(torch.float64)

    for emb in plan.embeddings:
        w = emb.weight.data
        emb.weight.data = (w.to(torch.float64) @ qd).to(w.dtype)

    for lin in plan.input_linears:
        w = lin.weight.data
        lin.weight.data = (w.to(torch.float64) @ qd).to(w.dtype)

    for lin in plan.output_linears:
        w = lin.weight.data
        lin.weight.data = (qd.t() @ w.to(torch.float64)).to(w.dtype)
        if lin.bias is not None:
            b = lin.bias.data
            lin.bias.data = (b.to(torch.float64) @ qd).to(b.dtype)


@torch.no_grad()
def rotate_value_heads(
    v_proj: nn.Linear,
    out_proj: nn.Linear,
    n_heads: int,
    head_dim: int,
    h: torch.Tensor,
    v_offset: Optional[int] = None,
) -> None:
    """Rotate each attention head's value subspace (QuaRot's R3).

    ``A @ (V H) = (A @ V) H``, so an orthogonal ``H`` applied per head inside V
    is undone by the same block-diagonal rotation on the out-projection's
    input.  This is the rotation that matters most for IA-AQ, because V is the
    tensor IA-AQ quantizes: after it, the value matrix has no outlier channel
    left for the shared per-feature scale to accommodate.

    ``v_offset`` is the column at which V starts inside a fused QKV projection.
    """
    hd = h.to(torch.float64)
    if h.shape != (head_dim, head_dim):
        raise ValueError(f"head rotation must be {head_dim}x{head_dim}, got {h.shape}")

    w = v_proj.weight.data
    lo = 0 if v_offset is None else v_offset
    hi = lo + n_heads * head_dim
    block = w[lo:hi, :].to(torch.float64).reshape(n_heads, head_dim, -1)
    block = torch.einsum("ij,hjk->hik", hd.t(), block)
    w[lo:hi, :] = block.reshape(n_heads * head_dim, -1).to(w.dtype)
    if v_proj.bias is not None:
        b = v_proj.bias.data[lo:hi].to(torch.float64).reshape(n_heads, head_dim)
        v_proj.bias.data[lo:hi] = (b @ hd).reshape(-1).to(v_proj.bias.dtype)

    wo = out_proj.weight.data
    blk = wo.to(torch.float64).reshape(-1, n_heads, head_dim)
    blk = torch.einsum("onh,hk->onk", blk, hd)
    out_proj.weight.data = blk.reshape(wo.shape).to(wo.dtype)


@torch.no_grad()
def rotate_qk(
    q: torch.Tensor, k: torch.Tensor, h: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """QuaRot's R4: head-wise rotation of Q and K, applied AFTER RoPE.

    Attention scores are exactly preserved, because the same orthogonal H hits
    both sides:  (QH)(KH)ᵀ = Q H Hᵀ Kᵀ = Q Kᵀ.

    It must come after the rotary embedding: RoPE rotates coordinate pairs
    within a head, and a Hadamard mixes all coordinates of the head, so the two
    do not commute and R4 cannot be folded into the projection weights.

    Why it looks different for a diffusion LM
    -----------------------------------------
    In an autoregressive model R4 exists to make the *KV cache* quantizable --
    that cache is the memory bottleneck at long context.  Vanilla LLaDA has no
    KV cache at all: attention is bidirectional and every token may change at
    the next denoising step, so keys and values are recomputed from scratch.
    Here R4 buys the other half of its original benefit instead -- it makes Q
    and K themselves quantizable, so the score matmul can run in INT4.

    The cache motivation returns for block-wise decoders (Fast-dLLM's
    approximate KV cache, LLaDA2.0's cache reuse), where R4 applies for exactly
    the reason QuaRot introduced it.
    """
    hd = h.to(q.dtype)
    if h.shape[0] != q.shape[-1]:
        raise ValueError(
            f"R4 rotation is {tuple(h.shape)} but head_dim is {q.shape[-1]}"
        )
    return q @ hd, k @ hd.to(k.dtype)


class QKRotation(nn.Module):
    """Holds the R4 matrix so a patched attention can call it."""

    def __init__(self, head_dim: int, seed: int = 0, device=None):
        super().__init__()
        self.register_buffer(
            "h", random_hadamard_matrix(head_dim, device=device, seed=seed),
            persistent=False,
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        return rotate_qk(q, k, self.h)


class OnlineHadamard(nn.Module):
    """Applies a Hadamard to a linear's input at runtime (QuaRot's R2).

    The MLP down-projection cannot have its rotation fused, because the
    elementwise activation in front of it does not commute with a rotation.
    QuaRot pays for an online transform instead -- O(d log d), no matrix.
    """

    def __init__(self, inner: nn.Module, name: str = "inner"):
        super().__init__()
        # Keep the wrapped module under its ORIGINAL attribute name. Every
        # downstream lookup -- CGQ's sequential groups, the skip patterns, the
        # value-projection search -- matches on the leaf name, and renaming
        # `ff_out` to `inner` silently drops it out of all of them.
        self._inner_name = name
        setattr(self, name, inner)

    @property
    def wrapped(self) -> nn.Module:
        return getattr(self, self._inner_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wrapped(fast_hadamard(x))


@torch.no_grad()
def install_online_hadamard(down_proj: nn.Linear) -> None:
    """Fuse Hᵀ into the down-projection so the online H cancels it."""
    n = down_proj.in_features
    if hadamard_factors(n) is None:
        raise ValueError(
            f"down-projection width {n} has no Hadamard construction; "
            "skip the online rotation for this model"
        )
    h = hadamard_matrix(n, device=down_proj.weight.device, dtype=torch.float64)
    w = down_proj.weight.data
    down_proj.weight.data = (w.to(torch.float64) @ h).to(w.dtype)


__all__ = [
    "hadamard_matrix",
    "random_hadamard_matrix",
    "fast_hadamard",
    "hadamard_factors",
    "fuse_norm_into_linears",
    "RotationPlan",
    "rotate_residual_stream",
    "rotate_value_heads",
    "OnlineHadamard",
    "install_online_hadamard",
]
