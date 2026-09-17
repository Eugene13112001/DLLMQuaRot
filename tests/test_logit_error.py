"""The logit error has to survive the K-to-Q gain migration as a comparable number.

check_key_error's K-space error changes meaning once the loud channels move into the
queries. The logit error does not: the migration leaves q . k^T bit-identical, so the
denominator is the same model, and the numerator is what attention receives. These
tests pin the two properties the stage-2 tensor run relies on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_key_error import logit_rel_err  # noqa: E402
from dllmquant.cache import quantize_kv  # noqa: E402


def toy(seed=0):
    torch.manual_seed(seed)
    b, hq, hk, t, d = 1, 4, 2, 16, 8
    q = torch.randn(b, hq, t, d, dtype=torch.float64)
    k = torch.randn(b, hk, t, d, dtype=torch.float64)
    k[..., 3] *= 20.0          # a loud fixed channel, as the K-norm gain makes
    return q, k


def test_exact_key_gives_zero_error():
    q, k = toy()
    assert logit_rel_err(q, torch.zeros_like(k), k, None) == 0.0


def test_denominator_is_invariant_under_the_migration():
    q, k = toy()
    s = torch.ones(k.shape[-1], dtype=torch.float64)
    s[3] = 16.0
    err = 0.01 * torch.randn_like(k)
    before = logit_rel_err(q, err, k, None)
    # the same error in the migrated frame: K divided by s, Q multiplied by s
    after = logit_rel_err(q * s, err / s, k / s, None)
    assert abs(before - after) < 1e-12


def test_per_channel_scale_does_not_notice_a_power_of_two_migration():
    q, k = toy()
    s = torch.ones(k.shape[-1], dtype=torch.float64)
    s[3] = 16.0
    ks = k / s
    e0 = quantize_kv(k.float(), 2, 128, axis="token").double() - k
    e1 = quantize_kv(ks.float(), 2, 128, axis="token").double() - ks
    a = logit_rel_err(q, e0, k, None)
    b = logit_rel_err(q * s, e1, ks, None)
    assert abs(a - b) / a < 1e-5


def test_mask_restricts_the_positions():
    q, k = toy()
    err = 0.01 * torch.randn_like(k)
    t = k.shape[-2]
    allowed = torch.tril(torch.ones(t, t, dtype=torch.float64))
    full = logit_rel_err(q, err, k, None)
    masked = logit_rel_err(q, err, k, allowed)
    assert masked != full
    zero = torch.zeros(t, t, dtype=torch.float64)
    assert logit_rel_err(q, err, k, zero) == 0.0
