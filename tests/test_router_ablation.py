"""Quantizing the router, so the exclusion can be a finding and not a policy.

The router is kept in FP16 by name, on the argument that its error changes
*which expert runs* rather than how accurately one runs. The argument is
good; it has never been checked. These tests cover the mechanics that make
checking it safe -- above all that the model is left exactly as it was found,
since the measurement runs many widths in a row against one loaded model and
a leaked perturbation would contaminate every row after it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_spec = importlib.util.spec_from_file_location(
    "check_router", Path(__file__).resolve().parents[1] / "scripts" / "check_router.py"
)
check_router = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_router)


class _Router(nn.Module):
    """A bare Parameter with F.linear around it, as LLaDA2.0's gate is."""

    def __init__(self, n_experts=8, hidden=16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_experts, hidden))
        self.num_experts = n_experts
        self.top_k = 2


def _routers(n=3):
    torch.manual_seed(0)
    return [_Router() for _ in range(n)]


def test_sixteen_bits_leaves_the_router_untouched():
    """The control row of the report depends on this being exactly true."""
    routers = _routers()
    before = [r.weight.data.clone() for r in routers]

    check_router.quantize_routers(routers, bits=16, group_size=-1)

    for r, w in zip(routers, before):
        assert torch.equal(r.weight.data, w)


def test_four_bits_visibly_changes_it():
    routers = _routers()
    before = [r.weight.data.clone() for r in routers]

    check_router.quantize_routers(routers, bits=4, group_size=-1)

    for r, w in zip(routers, before):
        assert not torch.allclose(r.weight.data, w)
        assert torch.isfinite(r.weight.data).all()


def test_the_model_is_handed_back_exactly_as_it_was():
    """Many widths run in a row against one loaded model. A perturbation that
    survives its own row silently corrupts every row after it, and the report
    would still look perfectly reasonable."""
    routers = _routers()
    before = [r.weight.data.clone() for r in routers]

    for bits in (8, 4, 2):
        originals = check_router.quantize_routers(routers, bits, group_size=-1)
        check_router.restore_routers(routers, originals)

    for r, w in zip(routers, before):
        assert torch.equal(r.weight.data, w), "a width leaked into the next row"


def test_restoring_returns_the_originals_not_copies_of_the_damage():
    routers = _routers(n=1)
    original = routers[0].weight.data.clone()

    saved = check_router.quantize_routers(routers, bits=2, group_size=-1)
    assert torch.equal(saved[0], original), "the saved tensor was taken too late"

    check_router.restore_routers(routers, saved)
    assert torch.equal(routers[0].weight.data, original)


def test_grouping_is_honoured():
    """Finer groups must hurt less; if the flag did nothing, they would tie."""
    torch.manual_seed(0)
    coarse, fine = _routers(n=1), _routers(n=1)
    original = coarse[0].weight.data.clone()

    check_router.quantize_routers(coarse, bits=3, group_size=-1)
    check_router.quantize_routers(fine, bits=3, group_size=4)

    err_coarse = (coarse[0].weight.data - original).abs().mean()
    err_fine = (fine[0].weight.data - original).abs().mean()
    assert err_fine < err_coarse
