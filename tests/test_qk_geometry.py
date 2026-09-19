"""The isotropic-noise factor has to predict the logit error of isotropic key noise.

check_key_error's qk_geometry is the proposed explanation for why a rotated per-token
key cache fails where the plain one is rescued: rotation makes the error isotropic, and
isotropic error hurts the logits in proportion to this factor. These tests pin that the
factor means what it claims before any checkpoint is read with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_key_error import logit_rel_err, qk_geometry, split_logit_err  # noqa: E402


def toy(seed=0, loud=20.0):
    torch.manual_seed(seed)
    q = torch.randn(1, 4, 24, 16, dtype=torch.float64)
    k = torch.randn(1, 2, 24, 16, dtype=torch.float64)
    k[..., 3] *= loud
    return q, k


def isotropic_error(k, sigma, seed=1):
    g = torch.Generator().manual_seed(seed)
    e = torch.randn(k.shape, generator=g, dtype=torch.float64)
    return e * sigma * k.norm(dim=-1, keepdim=True) / k.shape[-1] ** 0.5


def test_isotropic_error_follows_the_factor():
    q, k = toy()
    _, iso = qk_geometry(q, k, None)
    sigma = 0.1
    trials = [logit_rel_err(q, isotropic_error(k, sigma, s), k, None) for s in range(40)]
    measured = (sum(t * t for t in trials) / len(trials)) ** 0.5
    predicted = sigma * iso / k.shape[-1] ** 0.5
    assert measured == pytest.approx(predicted, rel=0.15)


def test_migration_keeps_the_logits_and_moves_the_factor_either_way():
    # Whether moving loudness into Q raises the factor is an empirical question, not a
    # theorem: it rises when Q was already loud where K was, and falls when K alone was.
    # That is exactly why it has to be measured on the checkpoint.
    q, k = toy()
    s = torch.ones(16, dtype=torch.float64)
    s[3] = 16.0
    kk = lambda t: t.repeat_interleave(2, 1).transpose(-1, -2)
    assert torch.allclose(q @ kk(k), (q * s) @ kk(k / s))
    _, iso_k_loud = qk_geometry(q, k, None)
    _, iso_moved = qk_geometry(q * s, k / s, None)
    assert iso_moved < iso_k_loud          # K alone loud: moving it balances the pair
    q2 = q.clone()
    q2[..., 3] *= 20.0                      # Q already loud in the same channel
    _, iso_both = qk_geometry(q2, k, None)
    _, iso_both_moved = qk_geometry(q2 * s, k / s, None)
    assert iso_both_moved > iso_both


def test_cosine_of_parallel_vectors_is_one():
    q = torch.ones(1, 1, 3, 4, dtype=torch.float64)
    k = 2 * torch.ones(1, 1, 3, 4, dtype=torch.float64)
    c, iso = qk_geometry(q, k, None)
    assert c == pytest.approx(1.0) and iso == pytest.approx(1.0)


def test_split_puts_all_error_where_it_is():
    q, k = toy()
    err = torch.zeros_like(k)
    err[..., 3] = 0.5
    loud = torch.zeros(16, dtype=torch.bool)
    loud[3] = True
    l_loud, l_rest = split_logit_err(q, err, k, None, loud)
    assert l_rest == 0.0
    assert l_loud == pytest.approx(logit_rel_err(q, err, k, None))
