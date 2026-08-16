"""Scales that live in the model instead of beside the cache.

The claim under test is arithmetic before it is empirical: a dynamic scale is
computed from the tensor it stores, so it must be stored with it, and that is
the quarter-bit between "four bits" and 4.25. A calibrated scale is a constant
of the model and costs the cache nothing -- *if* one canvas's scales are still
right on the next canvas.

These tests cover the parts of that which can be settled without the
checkpoint: the accounting, the survey statistics that decide whether the
sweep is worth running, and the wiring, end to end through the stand-in model,
so that a crash costs seconds here rather than an hour of a shared node.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import test_block_cache as tbc  # noqa: E402  the stand-in model lives there

from dllmquant.cache import (  # noqa: E402
    BlockKVCache,
    KVCacheConfig,
    StaticScaleBook,
)
from dllmquant.models.llada2_local import (  # noqa: E402
    logits_for_window,
    refresh_prefix,
)

_spec = importlib.util.spec_from_file_location(
    "check_static_scales",
    Path(__file__).resolve().parents[1] / "scripts" / "check_static_scales.py",
)
css = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(css)


# ---------------------------------------------------------------- accounting


def test_the_overhead_is_counted_from_the_shape_not_the_group_size():
    """224 tokens in groups of 128 is two groups, not one and three quarters."""
    assert css.overhead_bits("token", 128, 224, 128) == pytest.approx(0.2857, abs=1e-4)
    assert css.overhead_bits("token", 224, 224, 128) == pytest.approx(0.1429, abs=1e-4)
    # Along channels a group of 128 covers the whole head, so one scale per
    # token per head, whatever the sequence length.
    assert css.overhead_bits("channel", 128, 224, 128) == pytest.approx(0.25)
    assert css.overhead_bits("channel", 128, 448, 128) == pytest.approx(0.25)


def test_the_measured_champion_is_not_a_four_bit_cache():
    """The row this project reports as 3.76x, priced honestly."""
    k = 4 + css.overhead_bits("token", 128, 224, 128)
    v = 4 + css.overhead_bits("channel", 128, 224, 128)
    mean, compression = css.pair_cost(k, v)
    assert mean == pytest.approx(4.27, abs=0.01)
    assert compression == pytest.approx(3.75, abs=0.01)

    # Which is the whole point of calibrating: the same four bits, and the
    # scales no longer travel with the cache.
    assert css.pair_cost(4.0, 4.0) == (4.0, 4.0)


# ------------------------------------------------------------------- survey


def test_an_unmoving_scale_has_a_spread_of_one():
    same = [torch.tensor([1.0, 5.0, 0.2])] * 3
    med, p90 = css.spread(same)
    assert med == pytest.approx(1.0) and p90 == pytest.approx(1.0)

    # And the statistic is multiplicative: a channel that doubles reads x2
    # whether it sits at 0.1 or at 100.
    doubling = [torch.tensor([0.1, 100.0]), torch.tensor([0.2, 200.0])]
    assert css.spread(doubling)[0] == pytest.approx(2.0)


def test_fixed_outlier_channels_are_visible_against_chance():
    """The premise the token axis won on, stated as a measurement.

    If the fat channels are the same ones on every canvas a per-channel scale
    is a property of the model; if they are not, both this experiment and the
    axis result are accidents of one canvas.
    """
    torch.manual_seed(0)
    fixed, wandering = [], []
    for _ in range(4):
        base = torch.rand(512) + 0.5
        fat = base.clone()
        fat[[3, 17, 64, 129, 200, 301, 400, 501]] += 20.0
        fixed.append(fat)
        loose = base.clone()
        loose[torch.randperm(512)[:8]] += 20.0
        wandering.append(loose)

    stable, chance = css.top_channel_overlap(fixed)
    moving, _ = css.top_channel_overlap(wandering)
    assert stable == pytest.approx(1.0)
    assert moving < 0.3
    assert chance < 0.05


def test_the_split_does_not_read_sample_size_as_movement():
    """max/min grows with the number of draws; a variance split does not.

    This is the bug the first survey shipped with: 28 canvases against 2
    mask-ratio buckets, compared directly, and the same distribution on both
    axes reads x1.84 against x1.20 -- which was then reported as "the movement
    is not the mask ratio".
    """
    torch.manual_seed(0)
    many = [torch.exp(0.15 * torch.randn(64)) for _ in range(28)]
    few = [torch.exp(0.15 * torch.randn(64)) for _ in range(3)]
    assert css.spread(many)[0] > 1.4 * css.spread(few)[0]

    # Identical distributions in every bucket: the split must attribute
    # almost nothing to the bucket, whatever the group sizes.
    groups = [[torch.exp(0.15 * torch.randn(64)) for _ in range(n)]
              for n in (28, 28, 28)]
    within, between, share = css.variance_split(groups)
    assert within == pytest.approx(1.16, abs=0.06)
    assert between < 1.05 and share < 0.15


def test_the_split_finds_a_bucket_effect_when_there_is_one():
    torch.manual_seed(1)
    base = torch.rand(64) + 0.5
    groups = []
    for gain in (1.0, 5.0):
        groups.append([base * gain * torch.exp(0.02 * torch.randn(64))
                       for _ in range(6)])

    within, between, share = css.variance_split(groups)
    assert within < 1.05
    assert between > 3.0
    assert share > 0.9


def test_the_survey_prints_a_verdict_it_can_support(capsys):
    """A scale that does not transfer must be reported as such, not smoothed."""
    torch.manual_seed(1)
    scales = {}
    for layer in (0, 1):
        base = torch.rand(64) + 0.5          # this layer's channels, fixed
        for ratio in (0.0, 1.0):
            for i in range(6):
                # Barely moves between canvases, moves a lot with the ratio.
                obs = (base * (1.0 + 0.005 * i)
                       * (1.0 if ratio == 0.0 else 5.0))
                scales.setdefault((ratio, "key", layer), []).append(obs)

    css.run_survey(scales, [0.0, 1.0], ("key",))
    out = capsys.readouterr().out
    assert "from canvas to canvas" in out
    assert "most of what does move is the mask ratio" in out


# --------------------------------------------------------------- end to end


def _calibrated_book(model, states, x, prefix_len, bits, scale=1.0):
    """Run one forward through the stand-in and calibrate on what it wrote."""
    sink = {}
    tap = css.TapCache(len(model.model.layers), sink)
    tap.mask_ratio = 0.0
    for state in states:
        state.cache = tap
    refresh_prefix(model, states, x, prefix_len, tbc.BLOCK)

    book = StaticScaleBook(bits=bits, buckets=(0.0,))
    for layer, (k, v) in sink.items():
        book.observe(layer, "key", k * scale, 0.0)
        book.observe(layer, "value", v * scale, 0.0)
    return book.freeze()


def _window(model, states, cache, x, prefix_len, mask_ratio=0.0):
    cache.mask_ratio = mask_ratio
    for state in states:
        state.cache = cache
    refresh_prefix(model, states, x, prefix_len, tbc.BLOCK)
    return logits_for_window(model, states, x, lo=prefix_len, hi=x.shape[-1],
                             block_length=tbc.BLOCK)


def test_a_book_calibrated_on_this_canvas_is_the_dynamic_quantizer():
    """End to end, through attention, the two paths must coincide exactly.

    Same granularity, same affine arithmetic, same tensor -- so any difference
    in the logits would be a difference in code, and every later comparison
    between static and dynamic would be measuring it instead of the thing.
    """
    model, _, states = tbc._setup(bits=16)
    torch.manual_seed(0)
    x = torch.randint(0, tbc.VOCAB, (1, 12))
    prefix_len = 8
    n_layers = len(model.model.layers)

    book = _calibrated_book(model, states, x, prefix_len, bits=4)
    static = BlockKVCache(
        KVCacheConfig(enabled=True, decoded_bits=4, masked_bits=4,
                      scale_book=book),
        n_layers,
    )
    dynamic = BlockKVCache(
        KVCacheConfig(enabled=True, decoded_bits=4, masked_bits=4,
                      group_size=prefix_len, key_axis="token",
                      value_axis="token"),
        n_layers,
    )

    assert torch.equal(
        _window(model, states, static, x, prefix_len),
        _window(model, states, dynamic, x, prefix_len),
    )


def test_scales_from_the_wrong_distribution_are_visibly_worse():
    """And the wiring reports why: a book too wide resolves nothing."""
    model, _, states = tbc._setup(bits=16)
    torch.manual_seed(0)
    x = torch.randint(0, tbc.VOCAB, (1, 12))
    prefix_len = 8
    n_layers = len(model.model.layers)

    exact = _window(model, states, BlockKVCache(
        KVCacheConfig(enabled=True, decoded_bits=16, masked_bits=16), n_layers,
    ), x, prefix_len)

    def err(book):
        cache = BlockKVCache(
            KVCacheConfig(enabled=True, decoded_bits=4, masked_bits=4,
                          scale_book=book),
            n_layers,
        )
        out = _window(model, states, cache, x, prefix_len)
        return float((out - exact).abs().mean()), cache.stats

    right, _ = err(_calibrated_book(model, states, x, prefix_len, bits=4))
    wrong, stats = err(
        _calibrated_book(model, states, x, prefix_len, bits=4, scale=30.0)
    )

    assert wrong > 5 * right
    # Thirty times too wide clips nothing at all -- which is exactly why the
    # rail counter cannot be the only thing watched.
    assert stats.clip_rate == 0.0


def test_the_bucket_follows_the_mask_ratio_through_the_model():
    model, _, states = tbc._setup(bits=16)
    torch.manual_seed(0)
    x = torch.randint(0, tbc.VOCAB, (1, 12))
    prefix_len, n_layers = 8, len(model.model.layers)

    sink = {}
    tap = css.TapCache(n_layers, sink)
    for state in states:
        state.cache = tap
    tap.mask_ratio = 0.0
    refresh_prefix(model, states, x, prefix_len, tbc.BLOCK)

    book = StaticScaleBook(bits=4, buckets=(0.0, 1.0))
    for layer, (k, v) in sink.items():
        book.observe(layer, "key", k, 0.0)
        book.observe(layer, "value", v, 0.0)
        book.observe(layer, "key", 40.0 * k, 1.0)
        book.observe(layer, "value", 40.0 * v, 1.0)
    book.freeze()

    exact = _window(model, states, BlockKVCache(
        KVCacheConfig(enabled=True, decoded_bits=16, masked_bits=16), n_layers,
    ), x, prefix_len)

    def err(mask_ratio):
        cache = BlockKVCache(
            KVCacheConfig(enabled=True, decoded_bits=4, masked_bits=4,
                          scale_book=book),
            n_layers,
        )
        out = _window(model, states, cache, x, prefix_len, mask_ratio)
        return float((out - exact).abs().mean())

    assert err(1.0) > 5 * err(0.0)
    assert book.fallbacks == 0

