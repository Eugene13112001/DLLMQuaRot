"""Measuring what a cache would cost, without building one.

The trick the drift script rests on is that staleness and rounding are both
computable from captured tensors: K at step t against K at t+d is the first,
K against its own quantized copy is the second. Nothing here needs a cache to
exist, which is what makes the measurement available on LLaDA-1.5 as well --
a model whose attention is fully bidirectional and which therefore cannot have
one.

These tests exercise the capture path and the block bookkeeping against the
tiny stand-in model. The model-specific half -- that the probe recovers K/V
correctly from a real checkpoint -- is what selfcheck's attention probe check
already covers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_pipeline_e2e import H, HEAD_DIM, TinyAdapter, _config

_spec = importlib.util.spec_from_file_location(
    "measure_drift", Path(__file__).resolve().parents[1] / "scripts" / "measure_drift.py"
)
measure_drift = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(measure_drift)


# ---------------------------------------------------------------- block math


def test_a_position_is_placed_in_the_block_that_contains_it():
    bounds = [(8, 12), (12, 16), (16, 20)]
    assert measure_drift.block_of(8, bounds) == 0
    assert measure_drift.block_of(11, bounds) == 0
    assert measure_drift.block_of(12, bounds) == 1
    assert measure_drift.block_of(19, bounds) == 2


def test_the_prompt_sits_before_every_decoding_block():
    """Positions ahead of the first block are prompt: closed from the start,
    and they must count as prefix rather than as an error."""
    bounds = [(8, 12), (12, 16)]
    assert measure_drift.block_of(0, bounds) == -1
    assert measure_drift.block_of(7, bounds) == -1
    assert measure_drift.block_of(0, bounds) < 0  # i.e. closed before block 0


# ------------------------------------------------------------------ capture


def _adapter():
    torch.manual_seed(0)
    adapter = TinyAdapter(_config())
    adapter.load()
    return adapter


def test_capture_returns_hidden_and_kv_for_every_requested_layer():
    adapter = _adapter()
    x = torch.randint(0, 10, (1, 12))
    layers = [0, 1]

    data, keep = measure_drift.capture(adapter, x, layers, max_positions=64)

    assert set(data) == {0, 1}
    for layer in layers:
        assert data[layer]["hidden"].shape[1] == len(keep)
        assert data[layer]["k"].shape == (1, H, len(keep), HEAD_DIM)
        assert data[layer]["v"].shape == data[layer]["k"].shape
        assert data[layer]["k"].dtype == torch.float32


def test_positions_are_subsampled_when_asked():
    adapter = _adapter()
    x = torch.randint(0, 10, (1, 40))

    data, keep = measure_drift.capture(adapter, x, [0], max_positions=8)

    assert len(keep) == 8
    assert data[0]["hidden"].shape[1] == 8
    assert keep[0] == 0 and keep[-1] == 39, "the ends must be kept"


def test_the_hooks_are_removed_again():
    """A capture that leaked hooks would slow every later forward and quietly
    accumulate tensors across the whole run."""
    adapter = _adapter()
    x = torch.randint(0, 10, (1, 8))
    measure_drift.capture(adapter, x, [0, 1], max_positions=8)

    for block in adapter.blocks:
        assert not block._forward_hooks, "a forward hook survived the capture"


# --------------------------------------------------------------- the metric


def test_an_unchanged_state_has_no_staleness():
    a = torch.randn(1, 2, 6, 4)
    assert measure_drift.rel(a, a) == 0.0


def test_the_metric_is_relative_to_the_reference_scale():
    """Absolute differences say nothing across layers whose activations differ
    in magnitude by an order of magnitude, which they do."""
    a = torch.randn(1, 2, 6, 4)
    small = measure_drift.rel(a + 0.1, a)
    scaled = measure_drift.rel(10 * a + 1.0, 10 * a)
    assert abs(small - scaled) < 1e-5


def test_rounding_is_visible_at_four_bits_and_absent_at_sixteen():
    from dllmquant.cache import quantize_kv

    torch.manual_seed(0)
    k = torch.randn(1, 2, 16, 8)
    assert measure_drift.rel(quantize_kv(k, 16, 8), k) == 0.0
    assert measure_drift.rel(quantize_kv(k, 4, 8), k) > 0.01
