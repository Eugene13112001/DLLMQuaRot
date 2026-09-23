"""The replacement cache must round what it stores and nothing else.

Fast-dLLM-v2 builds its own prefix cache inside the model, so the only way in is to
replace the class it constructs. These tests pin the two things that makes or breaks:
the stored entries are the quantized ones, and at 16 bits the store is the original,
so a lossless run is a real control rather than a differently-rounded one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dllmquant.models.fast_dllm_v2_local import (  # noqa: E402
    FDv2CacheStats, make_quantized_cache_class,
)


class FakeCache:
    """The parts of DynamicCache the vendored model uses."""

    def __init__(self):
        self.k, self.v = [], []

    def update(self, key, value, layer_idx, cache_kwargs=None):
        while len(self.k) <= layer_idx:
            self.k.append(None)
            self.v.append(None)
        self.k[layer_idx] = key if self.k[layer_idx] is None else torch.cat(
            [self.k[layer_idx], key], dim=-2)
        self.v[layer_idx] = value if self.v[layer_idx] is None else torch.cat(
            [self.v[layer_idx], value], dim=-2)
        return self.k[layer_idx], self.v[layer_idx]


def build(bits, axis="token"):
    stats = FDv2CacheStats(key_axis=axis, value_axis="channel")
    cls = make_quantized_cache_class(
        FakeCache, key_bits=bits, value_bits=bits, group_size=128,
        key_axis=axis, value_axis="channel", stats=stats)
    return cls(), stats


def kv(seed=0):
    torch.manual_seed(seed)
    k = torch.randn(1, 2, 8, 16)
    k[..., 3] *= 20.0
    return k, torch.randn(1, 2, 8, 16)


def test_sixteen_bits_stores_the_original():
    cache, stats = build(16)
    k, v = kv()
    ok, ov = cache.update(k, v, 0)
    assert torch.equal(ok, k) and torch.equal(ov, v)
    assert stats.writes == 1 and stats.entries == 8


def test_two_bits_stores_the_rounded_tensor():
    cache, _ = build(2)
    k, v = kv()
    ok, _ = cache.update(k, v, 0)
    assert not torch.equal(ok, k)
    assert ok.dtype == k.dtype
    # one scale per channel over 8 tokens: 2 heads x 16 channels = 32 groups, 4 levels each
    assert len(ok.unique()) <= 4 * 32


def test_the_axis_reaches_the_quantizer():
    k, v = kv()
    per_channel, _ = build(2, axis="token")
    per_token, _ = build(2, axis="channel")
    a, _ = per_channel.update(k.clone(), v.clone(), 0)
    b, _ = per_token.update(k.clone(), v.clone(), 0)
    assert not torch.equal(a, b)
    # The quiet channels are what separates them: a per-token scale has to span the loud
    # channel, and everything else in that token is rounded against it. (Total Frobenius
    # error does not show this on a toy -- it is dominated by the loud channel itself.)
    quiet = [c for c in range(16) if c != 3]
    err = lambda x: float((x[..., quiet] - k[..., quiet]).norm() / k[..., quiet].norm())
    assert err(a) < err(b)


def test_entries_accumulate_across_writes():
    cache, stats = build(4)
    for i in range(3):
        k, v = kv(i)
        cache.update(k, v, 0)
    assert stats.writes == 3 and stats.entries == 24
    assert "K along token" in stats.describe()


def test_named_layers_are_stored_exactly():
    """The per-layer movement profile puts the rotation's damage in two layers of 28.

    Whether those two carry the whole collapse is a question about them, so the cache
    has to be able to leave them alone -- exactly, and without touching the others.
    """
    stats = FDv2CacheStats(key_axis="token", value_axis="channel")
    cls = make_quantized_cache_class(
        FakeCache, key_bits=2, value_bits=2, group_size=128,
        key_axis="token", value_axis="channel", stats=stats,
        skip_layers=frozenset({0, 27}))
    cache = cls()
    k, v = kv()
    for layer in (0, 1, 27):
        ok, ov = cache.update(k, v, layer)
        if layer in (0, 27):
            assert torch.equal(ok, k) and torch.equal(ov, v)
        else:
            assert not torch.equal(ok, k)
    assert stats.skipped == 2 and stats.writes == 1
    assert "2 writes kept exact" in stats.describe()


def test_a_skipped_layer_is_exact_even_with_noise_on():
    stats = FDv2CacheStats(key_axis="token", value_axis="channel")
    cls = make_quantized_cache_class(
        FakeCache, key_bits=16, value_bits=16, group_size=128,
        key_axis="token", value_axis="channel", stats=stats,
        key_noise=0.1, skip_layers=frozenset({5}))
    cache = cls()
    k, v = kv()
    ok, _ = cache.update(k, v, 5)
    assert torch.equal(ok, k)
    ok, _ = cache.update(k, v, 6)
    assert not torch.equal(ok, k)
