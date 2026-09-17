from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from policy_table import mcnemar  # noqa: E402


def test_mcnemar_counts_and_exact_p():
    a = [True, True, False, False, True]
    b = [True, False, True, True, True]
    up, down, p = mcnemar(a, b)
    assert (up, down) == (2, 1)
    assert p == pytest.approx(1.0)


def test_mcnemar_identical_runs():
    a = [True, False] * 10
    assert mcnemar(a, a) == (0, 0, 1.0)


def test_mcnemar_one_sided_flood_is_significant():
    a = [False] * 20
    b = [True] * 20
    up, down, p = mcnemar(a, b)
    assert (up, down) == (20, 0)
    assert p == pytest.approx(2 / 2 ** 20)


def test_mcnemar_refuses_mismatched_lengths():
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])
