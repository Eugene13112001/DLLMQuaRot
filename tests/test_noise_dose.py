"""The matched dose is the whole point of the structureless control.

If the noise is not brought to the same centered logit error the two models carry at four
bits per token, comparing their answers under it says nothing. These tests pin the fit and
the guard that tells us when proportionality failed and the table must be read by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from noise_dose import fit, read  # noqa: E402


def dump(tmp_path, name, model, sigma, err, bits=16, group=128):
    d = {"config": {"model": model, "bits": [bits], "group_size": [group],
                    "key_noise": sigma, "logit_error": True},
         "errors": {f"Lc/{bits}/token/{group}": [[err, err]],
                    f"KL/{bits}/token/{group}": [[0.01, 0.01]]}}
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return str(p)


def test_reads_the_dose_and_the_error(tmp_path):
    r = read(dump(tmp_path, "kn", "org/LLaDA2.0-mini", 0.05, 0.17))
    assert r["model"] == "LLaDA2.0-mini" and r["sigma"] == 0.05
    assert r["err"] == pytest.approx(0.17) and r["kl"] == pytest.approx(0.01)


def test_a_run_without_noise_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="key-noise"):
        read(dump(tmp_path, "plain", "org/m", 0.0, 0.17))


def test_the_fit_recovers_a_proportional_sweep():
    points = [{"sigma": s, "err": 3.4 * s} for s in (0.02, 0.05, 0.1, 0.2)]
    c = fit(points)
    assert c == pytest.approx(3.4, rel=1e-6)
    assert 0.35 / c == pytest.approx(0.1029, rel=1e-3)


def test_the_fit_is_pulled_by_a_sweep_that_saturates():
    """A sigma whose error stops growing is the case the residual guard is for."""
    points = [{"sigma": 0.02, "err": 0.068}, {"sigma": 0.05, "err": 0.17},
              {"sigma": 0.1, "err": 0.30}, {"sigma": 0.2, "err": 0.40}]
    c = fit(points)
    worst = max(abs(p["err"] - c * p["sigma"]) / p["err"] for p in points)
    assert worst > 0.1
