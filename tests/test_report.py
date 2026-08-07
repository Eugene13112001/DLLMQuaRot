"""CSV output: the format that actually gets opened on a cluster."""

from __future__ import annotations

import csv

import pytest

from dllmquant.report import round_floats, sibling_csv, write_csv


def test_writes_a_readable_csv(tmp_path):
    rows = [
        {"recipe": "FP16", "top1": 1.0, "kl": 0.0},
        {"recipe": "RTN", "top1": 0.913, "kl": 0.0421},
    ]
    p = write_csv(tmp_path / "out" / "compare.csv", rows)

    assert p.exists()
    with open(p, encoding="utf-8", newline="") as f:
        got = list(csv.DictReader(f))
    assert [r["recipe"] for r in got] == ["FP16", "RTN"]
    assert float(got[1]["top1"]) == pytest.approx(0.913)


def test_missing_keys_become_empty_cells():
    """Rows differ: GSM8K is only present when it was actually run."""
    rows = [{"recipe": "a", "gsm8k": 0.5}, {"recipe": "b"}]
    import tempfile
    import pathlib

    with tempfile.TemporaryDirectory() as d:
        p = write_csv(pathlib.Path(d) / "x.csv", rows)
        with open(p, encoding="utf-8", newline="") as f:
            got = list(csv.DictReader(f))
    assert got[1]["gsm8k"] == ""


def test_column_order_is_stable_across_runs(tmp_path):
    """Otherwise diffing two runs is noise."""
    rows = [{"b": 1, "a": 2}, {"c": 3, "a": 4}]
    p = write_csv(tmp_path / "a.csv", rows)
    with open(p, encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    assert header == ["b", "a", "c"]  # order of first appearance


def test_explicit_columns_win(tmp_path):
    rows = [{"a": 1, "b": 2, "junk": 3}]
    p = write_csv(tmp_path / "a.csv", rows, columns=["b", "a"])
    with open(p, encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    assert header == ["b", "a"]


def test_sibling_csv_replaces_the_extension():
    assert sibling_csv("out/compare.json").name == "compare.csv"
    assert str(sibling_csv("out/nested/r.json")).endswith("nested/r.csv".replace("/", __import__("os").sep))


def test_round_floats_leaves_other_types_alone():
    row = round_floats({"a": 1.23456789, "b": "text", "c": 5, "d": True}, digits=3)
    assert row == {"a": 1.235, "b": "text", "c": 5, "d": True}
