"""The statistics behind К3 and К4 must be right before they are read.

Synthetic dumps with known answers: a ratio the data pin exactly, a paired
design that cancels shared canvas noise, flips read from negative margins, and
exact permutation p-values checked against a count done by hand.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "stats_theses",
    Path(__file__).resolve().parents[1] / "scripts" / "stats_theses.py",
)
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)


def key_dump(path, k_ratio, v_ratio, canvases=12, layers=3, noise=0.3,
             seed=0, skip=False, shared=None):
    """A check_key_error dump whose axis ratios are exactly the given ones.

    Each canvas carries a level that varies a lot; the wrong axis is the
    right axis times the ratio, so the ratio of means is pinned whatever the
    levels, and only the level moves between draws.
    """
    rng = random.Random(seed)
    levels = shared or [[1 + noise * rng.random() for _ in range(layers)]
                        for _ in range(canvases)]
    errors = {}
    for side, r in (("K", k_ratio), ("V", v_ratio)):
        errors[f"{side}/4/token/128"] = levels
        errors[f"{side}/4/channel/128"] = [[r * x for x in row] for row in levels]
    d = {"config": {"model": "m/x", "model_type": "llada", "seq_len": 256,
                    "samples": canvases, "mask_ratio": 0.5,
                    "layers": list(range(layers)), "bits": [4],
                    "group_size": [128], "skip_qk_norm": skip},
         "shape": "canvas x layer", "errors": errors,
         "crest": {"K": levels, "V": levels}}
    path.write_text(json.dumps(d))
    return levels


def test_ratio_of_means_is_the_printed_quantity():
    num = [[2.0, 4.0], [6.0, 8.0]]
    den = [[1.0, 1.0], [1.0, 1.0]]
    assert st.ratio_of_means(num, den, [0, 1]) == pytest.approx(20 / 4)
    assert st.ratio_of_means(num, den, [1, 1]) == pytest.approx(28 / 4)


def test_pinned_ratio_has_a_zero_width_interval():
    rng = random.Random(1)
    levels = [[1 + rng.random() for _ in range(3)] for _ in range(10)]
    d = {"errors": {"K/4/channel/128": [[4.5 * x for x in r] for r in levels],
                    "K/4/token/128": levels}}
    est, draws = st.axis_draws(d, "K", 4, 128, 500, rng)
    assert est == pytest.approx(4.5)
    assert min(draws) == pytest.approx(4.5) and max(draws) == pytest.approx(4.5)


def test_axis_cross_model_separates_k_and_not_v(tmp_path, capsys):
    key_dump(tmp_path / "a.json", 4.5, 1.2, seed=1)
    key_dump(tmp_path / "b.json", 2.0, 1.2, seed=2)
    ns = type("A", (), dict(a=str(tmp_path / "a.json"), b=str(tmp_path / "b.json"),
                            paired=False, bits=4, group=128, boot=300, seed=0))
    assert st.cmd_axis(ns) == 0
    out = capsys.readouterr().out
    assert "K ratio of ratios  2.25" in out
    assert "V ratio of ratios  1.00" in out


def test_paired_refuses_different_canvases(tmp_path):
    key_dump(tmp_path / "a.json", 4.5, 1.2, canvases=12)
    key_dump(tmp_path / "b.json", 2.0, 1.2, canvases=10, skip=True)
    ns = type("A", (), dict(a=str(tmp_path / "a.json"), b=str(tmp_path / "b.json"),
                            paired=True, bits=4, group=128, boot=10, seed=0))
    with pytest.raises(SystemExit, match="samples"):
        st.cmd_axis(ns)


def test_sign_flip_exact_against_hand_count():
    # Three positive pairs of equal size: of eight sign patterns only
    # all-plus and all-minus reach |sum| = 3.
    p, exact = st.sign_flip_p([1.0, 1.0, 1.0])
    assert exact and p == pytest.approx(2 / 8)
    p, _ = st.sign_flip_p([1.0, -1.0])
    assert p == pytest.approx(1.0)


def test_relabel_exact_against_hand_count():
    # Two against two, fully separated: 2 of C(4,2)=6 splits are as extreme.
    p, exact = st.relabel_p([1.0, 1.0], [0.0, 0.0])
    assert exact and p == pytest.approx(2 / 6)


def test_flips_are_negative_margins_counted_per_prompt(tmp_path, capsys):
    def dump(path, flips_per_prompt, n=100, model="m/a"):
        cells = {}
        for i, f in enumerate(flips_per_prompt):
            for pos in range(n):
                cells[f"{i}:{pos}"] = [-0.5 if pos < f else 1.0, 0.9, 1.0]
        path.write_text(json.dumps({
            "config": {"model": model, "samples": len(flips_per_prompt),
                       "prompt_tokens": 32},
            "fields": ["margin", "top1_prob", "own_top2_gap"],
            "margin_def": "signed_reference_ordering",
            "cells": {"3/every_n:1": cells}}))

    dump(tmp_path / "a.json", [5, 6, 4, 5, 7, 5, 6, 4])
    dump(tmp_path / "b.json", [0, 1, 0, 0, 1, 0, 0, 1], model="m/b")
    flips, total = st.per_prompt(st.load_margins(str(tmp_path / "a.json")),
                                 "3/every_n:1")
    assert flips == [5, 6, 4, 5, 7, 5, 6, 4] and total == [100] * 8
    ns = type("A", (), dict(a=str(tmp_path / "a.json"), b=str(tmp_path / "b.json"),
                            cell="3/every_n:1", boot=300, seed=0))
    assert st.cmd_floor(ns) == 0
    out = capsys.readouterr().out
    assert "floor 5.25%" in out and "floor 0.38%" in out
    # Every pair points the same way, so the exact paired p is its floor.
    assert "p = 0.0078 (exact" in out


def test_unsigned_margins_are_refused(tmp_path):
    (tmp_path / "old.json").write_text(json.dumps({"cells": {}, "config": {}}))
    with pytest.raises(SystemExit, match="signed"):
        st.load_margins(str(tmp_path / "old.json"))


def test_rotate_reads_all_four_cells_and_compares_models(tmp_path, capsys):
    def dump(path, quarot_over_channel, model):
        levels = [[1.0 + 0.1 * c + 0.01 * l for l in range(3)] for c in range(10)]
        errors = {}
        for side in ("K", "V"):
            errors[f"{side}/4/token/128"] = levels
            errors[f"{side}/4/channel/128"] = [[3 * x for x in r] for r in levels]
            errors[f"{side}/4/channel+rot/128"] = [[quarot_over_channel * x for x in r] for r in levels]
            errors[f"{side}/4/token+rot/128"] = [[1.5 * x for x in r] for r in levels]
        path.write_text(json.dumps({
            "config": {"model": model, "samples": 10, "mask_ratio": 0.5, "layers": [0, 1, 2],
                       "bits": [4], "group_size": [128], "skip_qk_norm": False, "rotate": True,
                       "seq_len": 256, "model_type": "x"},
            "shape": "canvas x layer", "errors": errors, "crest": {}}))

    dump(tmp_path / "a.json", 2.0, "m/qknorm")
    dump(tmp_path / "b.json", 1.25, "m/plain")
    ns = type("A", (), dict(dumps=[str(tmp_path / "a.json"), str(tmp_path / "b.json")],
                            bits=4, group=128, boot=200, seed=0))
    assert st.cmd_rotate(ns) == 0
    out = capsys.readouterr().out
    assert "K QuaRot / per-channel scale  2.00x" in out
    assert "K QuaRot / per-channel scale  1.25x" in out
    assert "R4 improves the per-token scale 1.50x" in out
    assert "\n  1.60  95% CI" in out


def test_rotate_refuses_dump_without_r4(tmp_path):
    key_dump(tmp_path / "a.json", 4.5, 1.2)
    ns = type("A", (), dict(dumps=[str(tmp_path / "a.json")], bits=4, group=128, boot=10, seed=0))
    with pytest.raises(SystemExit, match="--rotate"):
        st.cmd_rotate(ns)


def test_rotation_rescues_the_per_token_scale_on_fixed_channel_outliers():
    """What R4 is for, on a stand-in: fixed-channel outliers ruin a per-token
    scale, and a Hadamard spreads them so the per-token scale recovers.

    Deliberately no claim about R4 against a per-channel scale. On Gaussian
    keys with three loud channels the rotation actually wins (0.068 against
    0.097), so which of the two is better is an empirical property of real
    keys -- the thing the --rotate dumps measure -- and not something a toy
    tensor can settle either way.
    """
    import torch
    from dllmquant.algos.quarot import random_hadamard_matrix
    from dllmquant.cache import quantize_kv

    g = torch.Generator().manual_seed(0)
    k = torch.randn(1, 4, 256, 64, generator=g)
    k[..., :3] *= 20.0
    h = random_hadamard_matrix(64, seed=0)
    err = lambda q, t: float((q - t).norm() / t.norm())
    kr = k @ h
    quarot = err(quantize_kv(kr, 4, 128, axis="channel"), kr)
    per_token = err(quantize_kv(k, 4, 128, axis="channel"), k)
    assert quarot < 0.6 * per_token
    # orthogonal: the error in the rotated frame is the error after undoing it
    back = quantize_kv(kr, 4, 128, axis="channel") @ h.T
    assert err(back, k) == pytest.approx(quarot, rel=1e-4)
