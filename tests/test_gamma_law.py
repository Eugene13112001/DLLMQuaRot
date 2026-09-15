"""The weight-law analysis must read the right cells and compute honest correlations."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "gamma_law", Path(__file__).resolve().parents[1] / "scripts" / "gamma_law.py")
gl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gl)


def test_spearman_and_ties():
    assert gl.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert gl.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert gl.ranks([5, 1, 5, 3]) == [3.5, 1.0, 3.5, 2.0]


def test_permutation_p_small_for_perfect_order_and_large_for_noise():
    xs = list(range(20))
    assert gl.perm_p(xs, [x * 2 for x in xs], n=2000) < 0.001
    assert gl.perm_p(xs, [(7 * x) % 20 for x in xs], n=2000) > 0.05


def test_pair_gains_combine_rotary_partners():
    g = [3.0, 1.0, 4.0, 1.0]            # rd = 4: pairs (0, 2) and (1, 3)
    assert gl.pair_gains(g, 4) == pytest.approx([5.0, math.sqrt(2)])


def test_constant_feature_gives_no_p_value():
    assert math.isnan(gl.perm_p([1.0] * 10, list(range(10)), n=100))


def test_features_flag_one_loud_channel():
    flat = gl.features([1.0] * 32)
    loud = gl.features([1.0] * 31 + [30.0])
    assert flat["max/median"] == pytest.approx(1.0)
    assert loud["max/median"] == pytest.approx(30.0)
    assert loud["top-share"] > flat["top-share"]


def _dump(path, layers, gains, axis_cost, nonorm=False, rd=64, head=128, canvases=4):
    errors = {}
    for part in (None, "rot", "pass"):
        suffix = "" if part is None else "@" + part
        tok = [[1.0] * len(layers) for _ in range(canvases)]
        ch = [[axis_cost[l] for l in range(len(layers))] for _ in range(canvases)]
        errors[f"K/4/token{suffix}/128"] = tok
        errors[f"K/4/channel{suffix}/128"] = ch
    path.write_text(json.dumps({
        "config": {"model": "m/x", "samples": canvases, "mask_ratio": 0.5, "layers": layers, "bits": [4],
                   "group_size": [128], "skip_qk_norm": nonorm, "split_rope": True, "rotate": False},
        "shape": "canvas x layer", "errors": errors, "crest": {},
        "gamma": {"k": gains, "q": gains}, "rotary_dim": [rd] * len(layers), "rope_applied": [True] * len(layers),
    }))


def test_run_reports_positive_law_on_synthetic_layers(tmp_path, capsys):
    layers = list(range(12))
    gains = []
    for l in layers:
        g = [1.0] * 128
        g[100] = 1.0 + 2.0 * l        # the loud unrotated channel grows with depth
        gains.append(g)
    cost = [1.5 + 0.3 * l for l in layers]
    _dump(tmp_path / "a.json", layers, gains, cost)
    _dump(tmp_path / "b.json", layers, gains, [2.0] * 12, nonorm=True)
    ns = type("A", (), dict(dump=str(tmp_path / "a.json"), nonorm=str(tmp_path / "b.json"),
                            bits=4, group=128, boot=200, perm=500))
    assert gl.run(ns) == 0
    out = capsys.readouterr().out
    assert "=== part RoPE leaves alone ===" in out
    assert "rho(max/median , axis added by norm  ) = +1.00" in out


def test_run_refuses_model_without_norm(tmp_path):
    layers = [0, 1]
    _dump(tmp_path / "a.json", layers, [None, None], [1.0, 1.0])
    ns = type("A", (), dict(dump=str(tmp_path / "a.json"), nonorm=None, bits=4, group=128, boot=10, perm=10))
    with pytest.raises(SystemExit, match="no K-norm gains"):
        gl.run(ns)
