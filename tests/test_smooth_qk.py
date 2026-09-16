"""Moving the K-norm gain into the Q-norm must leave attention exactly as it was.

The intervention is only a control if it is an identity on the model's own arithmetic. The
pairing constraint is what makes it one: RoPE rotates channel c with c + rd/2, and a diagonal
rescaling commutes with that rotation only when both channels of the pair carry the same factor.
The last test here is the counterexample -- an unconstrained factor does move the logits, and it
moves them more the further apart the two positions are.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dllmquant.algos import smooth_qk as sq  # noqa: E402


def rope(x: torch.Tensor, rd: int) -> torch.Tensor:
    """Llama-convention partial RoPE on [T, D]: first rd channels rotate, the rest pass."""
    t, d = x.shape
    half = rd // 2
    inv = 1.0 / (10000.0 ** (torch.arange(0, rd, 2, dtype=torch.float64) / rd))
    ang = torch.outer(torch.arange(t, dtype=torch.float64), inv)
    cos, sin = torch.cat((ang, ang), -1).cos(), torch.cat((ang, ang), -1).sin()
    head, tail = x[:, :rd], x[:, rd:]
    rot = torch.cat((-head[:, half:], head[:, :half]), dim=-1)
    return torch.cat((head * cos + rot * sin, tail), dim=-1)


def logits(q, k, gq, gk, rd):
    return rope(q * gq, rd) @ rope(k * gk, rd).T


@pytest.fixture
def toy():
    torch.manual_seed(0)
    d, rd, t = 16, 8, 12
    q = torch.randn(t, d, dtype=torch.float64)
    k = torch.randn(t, d, dtype=torch.float64)
    gq = torch.rand(d, dtype=torch.float64) + 0.5
    gk = torch.rand(d, dtype=torch.float64) + 0.5
    gk[3] = 9.0                      # a loud rotary channel
    gk[3 + rd // 2] = 0.4            # its rotary partner, deliberately different
    gk[13] = 7.0                     # a loud channel RoPE never touches
    return q, k, gq, gk, rd


def test_migration_leaves_every_logit_where_it_was(toy):
    q, k, gq, gk, rd = toy
    s = sq.gain_scales(gk, rd, alpha=1.0).double()
    before = logits(q, k, gq, gk, rd)
    after = logits(q, k, gq * s, gk / s, rd)
    assert torch.allclose(before, after, atol=1e-10)


def test_alpha_interpolates_and_zero_is_a_no_op(toy):
    _, _, _, gk, rd = toy
    assert torch.allclose(sq.gain_scales(gk, rd, alpha=0.0), torch.ones_like(gk))
    # the interpolation is exact before the power-of-two rounding; with it, half a step is
    # whatever the nearest exponent allows, so the property is checked where it is defined
    half = sq.gain_scales(gk, rd, alpha=0.5, pow2=False)
    full = sq.gain_scales(gk, rd, alpha=1.0, pow2=False)
    assert torch.allclose(half ** 2, full, atol=1e-10)


def test_full_migration_flattens_the_key_gains(toy):
    _, _, _, gk, rd = toy
    flat = gk / sq.gain_scales(gk, rd, alpha=1.0)
    spread = lambda g: float(g.abs().max() / g.abs().median())
    # A rotary pair can only be flattened to its own geometric mean, and the factor is rounded
    # to a power of two, so this is a coarse flattening by construction -- but the peak has to
    # come down a long way.
    assert spread(flat) < spread(gk) / 2


def test_power_of_two_factor_keeps_the_product_bit_exact_in_bfloat16(toy):
    _, _, gq, gk, rd = toy
    gq16, gk16 = gq.to(torch.bfloat16), gk.to(torch.bfloat16)
    s = sq.gain_scales(gk, rd, alpha=1.0).to(torch.bfloat16)
    moved = (gk16 / s).float() * (gq16 * s).float()
    assert torch.equal(moved, gk16.float() * gq16.float())
    # without the rounding the same migration loses the last mantissa bits
    s_raw = sq.gain_scales(gk, rd, alpha=1.0, pow2=False).to(torch.bfloat16)
    sloppy = (gk16 / s_raw).float() * (gq16 * s_raw).float()
    assert not torch.equal(sloppy, gk16.float() * gq16.float())


def test_every_factor_is_a_power_of_two(toy):
    _, _, _, gk, rd = toy
    s = sq.gain_scales(gk, rd, alpha=1.0).double()
    assert torch.allclose(torch.log2(s), torch.round(torch.log2(s)), atol=1e-12)


def test_pairs_must_share_the_factor_or_the_logits_move(toy):
    q, k, gq, gk, rd = toy
    s = gk.abs() / torch.exp(torch.log(gk.abs()).mean())   # per channel, pairing ignored
    before = logits(q, k, gq, gk, rd)
    after = logits(q, k, gq * s, gk / s, rd)
    assert not torch.allclose(before, after, atol=1e-6)
    # and the damage depends on the distance between the two positions, which is the silent
    # failure to avoid: it is not a constant offset one could notice on a single pair.
    diff = (after - before).abs()[0]
    assert float(diff.max() - diff.min()) > 1e-3


def test_odd_rotary_width_is_refused(toy):
    _, _, _, gk, _ = toy
    with pytest.raises(ValueError, match="odd"):
        sq.gain_scales(gk, rotary_dim=7, alpha=1.0)
    with pytest.raises(ValueError, match="alpha"):
        sq.gain_scales(gk, rotary_dim=8, alpha=1.5)
