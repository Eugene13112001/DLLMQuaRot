"""The two objections our own explanations face, made measurable.

KVTuner reports sparse-attention heads as the more robust to cache quantization, the
opposite sign to ours. The reconciliation on offer is the attention sink: if the first
few positions hold the mass, perturbing the distribution changes no order. And a
zero-mean perturbation of only the cached keys inflates their softmax weight by
exp(Var/2), which is a rival explanation for noise being worse than the quantizer.
These tests pin both measurements on cases where the answer is known by construction.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_key_error import sink_and_jensen  # noqa: E402


def toy(t=32, d=8, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(1, 4, t, d, dtype=torch.float64)
    k = torch.randn(1, 2, t, d, dtype=torch.float64)
    return q, k, 1.0 / math.sqrt(d)


def test_a_sink_is_detected_and_removed():
    """A key aligned with every query takes the mass; dropping it leaves a broad rest.

    Scaling an arbitrary key is not a sink: half the queries have a negative dot
    product with it, so it becomes the least attended for them. A sink is a direction
    every query shares, which is how the real one arises.
    """
    q, k, sc = toy()
    common = torch.ones(1, 1, 1, k.shape[-1], dtype=torch.float64)
    q = q + 3.0 * common                    # every query leans the same way
    k = k.clone()
    k[:, :, 0:1] = 8.0 * common             # position 0 leans there too
    mass, eff_nosink, _, _ = sink_and_jensen(q, k, None, sc, sink=1)
    assert mass > 0.8
    assert eff_nosink > 5.0                 # the rest is spread out


def test_without_a_sink_the_count_barely_moves():
    q, k, sc = toy()
    mass, eff_nosink, _, _ = sink_and_jensen(q, k, None, sc, sink=4)
    t = k.shape[-2]
    assert mass < 4.0 / t * 3               # no more than a fair share, roughly
    assert eff_nosink > 0.5 * t             # still broad


def test_the_sink_is_counted_among_allowed_positions_only():
    """Under a causal mask the first allowed key of a query is the query's own first."""
    q, k, sc = toy(t=16)
    tt = k.shape[-2]
    allowed = (torch.arange(tt)[:, None] >= torch.arange(tt)[None, :]).to(torch.float64)
    mass, eff, _, _ = sink_and_jensen(q, k, allowed, sc, sink=1)
    assert 0.0 < mass < 1.0
    assert eff >= 1.0


def test_jensen_term_is_quadratic_in_the_query_norm():
    """The term is ||q||^2 rms(k)^2 / 2d, so doubling the queries quadruples it."""
    q, k, sc = toy()
    _, _, j1, _ = sink_and_jensen(q, k, None, sc)
    _, _, j2, _ = sink_and_jensen(2 * q, k, None, sc)
    assert j2 == pytest.approx(4 * j1, rel=1e-6)


def test_jensen_term_matches_the_closed_form():
    q, k, sc = toy()
    _, _, j, _ = sink_and_jensen(q, k, None, sc)
    d = k.shape[-1]
    want = float(q.pow(2).sum(dim=-1).mean() * k.pow(2).mean() / (2 * d))
    assert j == pytest.approx(want, rel=1e-6)
