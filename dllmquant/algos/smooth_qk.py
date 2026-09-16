"""Move the K-norm gain into the Q-norm, without changing a single answer.

QK-Norm scales channel *c* of every key by a fixed gain ``gamma_K[c]`` and every query by
``gamma_Q[c]``. The attention logit is ``q . k``, so dividing the key gains by any positive
vector ``s`` and multiplying the query gains by the same ``s`` leaves every logit untouched --
and takes the loud fixed channels out of the tensor the cache stores. That is the intervention
the QK-Norm finding has been missing: the correlation says the gain is why the wrong key axis
costs so much on LLaDA2.0-mini, and this turns the gain off inside one model, with the model's
own outputs as the control.

**Rotary is the one real constraint.** RoPE rotates channel ``c`` together with ``c + rd/2`` by
an angle that depends on the position, and a diagonal rescaling commutes with that rotation only
when both channels of the pair carry the same factor. So ``s`` is forced to be constant on each
rotary pair; the channels RoPE leaves alone are free. Ignore this and the logits shift with
position -- silently, and only at long range.

The transform is the one QServe calls SmoothAttention and HiFA4 calls Smooth-QK. It is used
there to make quantization cheaper; here it is used as a control: if the model still answers
exactly as before and the two-bit collapse disappears, the gain was the cause.

``alpha`` interpolates: 0 leaves the model alone, 1 flattens the key gains completely (every
channel ends at the geometric mean of the pair-constrained gains), 0.5 splits the difference the
way SmoothQuant does.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn

from ..models.llada import _K_NORM_NAMES, _Q_NORM_NAMES
from ..models.base import find_submodule


def gain_scales(gamma_k: torch.Tensor, rotary_dim: int, alpha: float = 0.5,
                pow2: bool = True) -> torch.Tensor:
    """Per-channel factor to divide the key gains by (and multiply the query gains by).

    Constant on every rotary pair, geometric-mean-normalised so the overall scale of the
    logits' two factors does not drift, and raised to ``alpha`` to interpolate.

    ``pow2`` rounds the factor to a power of two, and that is what makes the transform exact
    rather than nearly exact: the gains live in bfloat16, which keeps eight mantissa bits, so an
    arbitrary factor loses about half a percent on each side and the product -- the only thing
    attention sees -- drifts with it. A power of two only moves the exponent, so dividing one
    gain and multiplying the other returns the identical product in any float format.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    g = gamma_k.detach().abs().double().clamp_min(1e-8)
    d = g.numel()
    rd = max(0, min(int(rotary_dim), d))
    if rd % 2:
        raise ValueError(f"rotary width {rd} is odd; RoPE pairs cannot be formed")
    paired = g.clone()
    if rd:
        half = rd // 2
        # Both channels of a pair must share the factor; their geometric mean keeps the
        # pair's product -- and so its energy -- where it was.
        pair = torch.sqrt(g[:half] * g[half:rd])
        paired[:half] = pair
        paired[half:rd] = pair
    s = (paired / torch.exp(torch.log(paired).mean())) ** alpha
    if pow2:
        s = torch.pow(2.0, torch.round(torch.log2(s)))
    return s.to(gamma_k.dtype)


def _norms(block: nn.Module) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
    return find_submodule(block, _Q_NORM_NAMES), find_submodule(block, _K_NORM_NAMES)


def migrate_qk_gains(adapter, alpha: float = 0.5, pow2: bool = True) -> Callable[[], None]:
    """Apply the migration to every block of a loaded adapter. Returns the undo."""
    rotary_dim = adapter._probe_rotary_dim() or adapter.head_dim
    saved: List[Tuple[nn.Parameter, torch.Tensor, nn.Parameter, torch.Tensor]] = []
    touched = 0
    for block in adapter.blocks:
        q_norm, k_norm = _norms(block)
        if q_norm is None or k_norm is None:
            raise RuntimeError(
                "this model has no QK-Norm on both Q and K, so there is no gain to move; "
                "the intervention is only defined where the norm exists")
        wq, wk = q_norm.weight, k_norm.weight
        if wq.shape != wk.shape or wk.numel() != adapter.head_dim:
            raise RuntimeError(
                f"expected per-head gains of width {adapter.head_dim}, got {tuple(wk.shape)} "
                "-- a norm over the whole projection would need a different pairing")
        s = gain_scales(wk.data, rotary_dim, alpha, pow2).to(wk.device)
        saved.append((wq, wq.data.clone(), wk, wk.data.clone()))
        with torch.no_grad():
            wk.data = wk.data / s
            wq.data = wq.data * s
        touched += 1

    def undo() -> None:
        with torch.no_grad():
            for wq, oq, wk, ok in saved:
                wq.data = oq
                wk.data = ok

    print(f"QK gains migrated on {touched} blocks (alpha={alpha}, rotary pairs held equal, "
          f"{'powers of two' if pow2 else 'exact factors'}): "
          "keys lose the fixed-channel peaks, queries take them, logits unchanged")
    return undo


def logit_drift(adapter, alpha: float = 0.5, seq_len: int = 64, seed: int = 0) -> float:
    """Largest relative change in one block's attention logits after the migration.

    The claim is exactness, so it has to be checked on the real checkpoint rather than argued.
    In bfloat16 the residue is rounding, not structure; anything beyond about 1e-2 means the
    pairing is wrong and the transform is not the identity it is sold as.
    """
    block = adapter.blocks[0]
    torch.manual_seed(seed)
    probe = adapter.make_probe(block)
    dev = next(adapter.model.parameters()).device
    x = torch.randint(0, 1000, (1, seq_len), device=dev)
    with probe, torch.no_grad():
        adapter.model(x, **adapter.forward_kwargs(x))
        before = probe.parts.attn_probs.detach().float().clone()
    undo = migrate_qk_gains(adapter, alpha)
    try:
        probe = adapter.make_probe(block)
        with probe, torch.no_grad():
            adapter.model(x, **adapter.forward_kwargs(x))
            after = probe.parts.attn_probs.detach().float()
        return float((after - before).abs().max() / before.abs().max().clamp_min(1e-9))
    finally:
        undo()
