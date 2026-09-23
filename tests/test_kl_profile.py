"""The mean over layers is not what accuracy follows; this reads the alternatives.

Three Fast-dLLM-v2 cells at mean KL 0.348, 0.351 and 0.355 score 43.5, 1.5 and 28.0. The
candidates are order statistics over layers, and reading them right matters more than
usual here, because a wrong axis name or group silently selects a different cell.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from kl_profile import cell, pct  # noqa: E402


def dump(tmp_path, name, rows, layers=(0, 1, 2, 3)):
    d = {"config": {"model": "org/m", "layers": list(layers)},
         "errors": {"KL/2/token/128": rows}}
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return str(p)


def test_averages_canvases_and_keeps_layers(tmp_path):
    # two canvases, four layers
    p = dump(tmp_path, "a", [[0.1, 0.2, 0.3, 0.4], [0.3, 0.2, 0.1, 0.0]])
    r = cell(f"{p}:2:token", "KL", 128)
    assert r["per_layer"] == pytest.approx([0.2, 0.2, 0.2, 0.2])
    assert r["layers"] == [0, 1, 2, 3]


def test_a_missing_cell_names_what_is_there(tmp_path):
    p = dump(tmp_path, "b", [[0.1, 0.1, 0.1, 0.1]])
    with pytest.raises(SystemExit, match="KL/2/token/128"):
        cell(f"{p}:2:channel", "KL", 128)


def test_a_concentrated_profile_is_visible_where_the_mean_is_not(tmp_path):
    """Two cells of equal mean: one flat, one with a single ruined layer."""
    flat = cell(f"{dump(tmp_path, 'flat', [[0.4, 0.4, 0.4, 0.4]])}:2:token", "KL", 128)
    spike = cell(f"{dump(tmp_path, 'spike', [[0.1, 0.1, 0.1, 1.3]])}:2:token", "KL", 128)
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    assert mean(flat["per_layer"]) == pytest.approx(mean(spike["per_layer"]))
    assert max(spike["per_layer"]) > 3 * max(flat["per_layer"])


def test_percentile_picks_a_measured_layer():
    v = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert pct(v, 0.9) in v and pct(v, 0.9) == 0.5
    assert pct(v, 0.0) == 0.1
