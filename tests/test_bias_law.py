from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bias_law import bias_features  # noqa: E402


def test_a_flat_bias_has_ratio_one():
    b = [1.0] * 16 * 2          # two heads of 16 channels
    f = bias_features(b, kv_heads=2)
    assert f["pair max/median"] == pytest.approx(1.0)
    assert f["max |b|"] == pytest.approx(1.0)


def test_one_loud_pair_raises_the_concentration():
    flat = [1.0] * 16
    loud = [1.0] * 16
    loud[2] = 20.0             # pairs with channel 2 + 8 = 10
    f_flat = bias_features(flat, kv_heads=1)
    f_loud = bias_features(loud, kv_heads=1)
    assert f_loud["pair max/median"] > 10 * f_flat["pair max/median"]
    assert f_loud["pair top-share"] > f_flat["pair top-share"]


def test_heads_are_averaged_not_pooled():
    head_a = [1.0] * 16
    head_b = [1.0] * 16
    head_b[0] = 30.0
    f = bias_features(head_a + head_b, kv_heads=2)
    single = bias_features(head_b, kv_heads=1)
    assert f["pair max/median"] == pytest.approx((1.0 + single["pair max/median"]) / 2)
