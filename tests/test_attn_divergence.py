"""What softmax does with the cache error, as opposed to how large the error is.

Thesis 4 is open because two models carry the same relative logit error at four bits
per token (0.351 and 0.346) and score 91.5 and 0.0. The relative error cannot tell
them apart by construction: softmax reads logits against their own spread. The obvious answer -- read
the error in units of the spread of the logits -- is no answer: the centered relative
error already is that ratio, and the first test below pins the identity. What is left
is how much attention *moves*, and the last test shows that is not a function of the
error either.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_key_error import attn_divergence, logit_rel_err  # noqa: E402


def toy(seed=0, t=16, d=8):
    torch.manual_seed(seed)
    q = torch.randn(1, 4, t, d, dtype=torch.float64)
    k = torch.randn(1, 2, t, d, dtype=torch.float64)
    return q, k, 1.0 / math.sqrt(d)


def test_exact_key_moves_nothing():
    q, k, sc = toy()
    kl, flip, eff, edev = attn_divergence(q, torch.zeros_like(k), k, None, sc)
    assert kl == 0.0 and flip == 0.0 and edev == 0.0
    assert 1.0 <= eff <= k.shape[-2]


def test_edev_is_the_centered_relative_error():
    """Not a new quantity: this is why the spread cannot explain thesis 4."""
    q, k, sc = toy()
    err = 0.05 * torch.randn_like(k)
    t = k.shape[-2]
    allowed = (torch.arange(t)[:, None] >= torch.arange(t)[None, :]).to(torch.float64)
    for al in (None, allowed):
        _, _, _, edev = attn_divergence(q, err, k, al, sc)
        assert abs(edev - logit_rel_err(q, err, k, al, center=True)) < 1e-12


def test_blind_to_a_component_every_key_shares():
    """A vector added to every key shifts each query's logits by a constant."""
    q, k, sc = toy()
    err = torch.zeros_like(k) + torch.randn(k.shape[-1], dtype=torch.float64)
    kl, flip, _, edev = attn_divergence(q, err, k, None, sc)
    assert kl < 1e-12 and flip == 0.0 and edev < 1e-9


def test_mask_restricts_the_keys():
    q, k, sc = toy()
    t = k.shape[-2]
    allowed = (torch.arange(t)[:, None] >= torch.arange(t)[None, :]).to(torch.float64)
    err = torch.zeros_like(k)
    err[:, :, -1] = 100.0          # only the last key, which early queries may not read
    kl, _, _, _ = attn_divergence(q, err, k, allowed, sc)
    full, _, _, _ = attn_divergence(q, err, k, None, sc)
    assert kl < full


def test_flips_are_counted_when_the_top_key_changes():
    q, k, sc = toy()
    kl0, flip0, _, _ = attn_divergence(q, torch.zeros_like(k), k, None, sc)
    kl, flip, _, edev = attn_divergence(q, 50.0 * torch.randn_like(k), k, None, sc)
    assert flip0 == 0.0 and flip > 0.5 and kl > 0.0 and edev > 1.0


def test_movement_is_not_a_function_of_the_error():
    """The reason the measure is worth taking: same centered error, different KL.

    One key made six times longer concentrates attention. Matching the two runs on
    the centered error -- the quantity at which LLaDA2.0-mini and Fast-dLLM-v2 are
    indistinguishable -- still leaves the peaked one moving further per unit error.
    """
    q, k, sc = toy()
    peaked = k.clone()
    peaked[:, :, 0] *= 6.0
    err = 0.05 * torch.randn_like(k)
    kl_p, _, eff_p, edev_p = attn_divergence(q, err, peaked, None, sc)
    # scale the flat run's error to the same centered error, then compare movement
    _, _, eff_f, edev_f = attn_divergence(q, err, k, None, sc)
    kl_f, _, _, edev_f2 = attn_divergence(q, err * (edev_p / edev_f), k, None, sc)
    assert abs(edev_f2 - edev_p) < 0.05 * edev_p
    assert eff_p < eff_f
    assert kl_p > 1.5 * kl_f
