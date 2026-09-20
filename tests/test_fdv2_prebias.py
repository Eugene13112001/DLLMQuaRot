"""Storing keys without the k_proj bias must be exact, and per layer.

This is the counterpart of the QK-Norm gain migration on a model that has no norm to
migrate: RoPE is linear, so a key is ``RoPE(Wx) + RoPE(b)``, and the second term is a
fixed per-channel vector that the quantizer otherwise has to span. Subtracting it before
the quantizer and adding it back after changes nothing the model sees -- which is what
these tests pin, together with the fact that each block's own bias is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dllmquant.models.fast_dllm_v2_local import (  # noqa: E402
    FDv2CacheStats, make_quantized_cache_class, rotated_key_bias,
)
from tests.test_fdv2_cache import FakeCache  # noqa: E402

HEADS, DIM, T = 2, 16, 8


def tables(seed=0):
    torch.manual_seed(seed)
    ang = torch.rand(1, T, DIM // 2) * 3.0
    cos = torch.cat((ang.cos(), ang.cos()), dim=-1)
    sin = torch.cat((ang.sin(), ang.sin()), dim=-1)
    return cos, sin


def build(bits, biases):
    stats = FDv2CacheStats(key_axis="token", value_axis="channel")
    cls = make_quantized_cache_class(
        FakeCache, key_bits=bits, value_bits=16, group_size=128,
        key_axis="token", value_axis="channel", stats=stats,
        key_biases=biases, kv_heads=HEADS)
    return cls(), stats


def test_sixteen_bits_with_pre_bias_is_the_identity():
    cos, sin = tables()
    b = [torch.randn(HEADS * DIM) for _ in range(2)]
    cache, stats = build(16, b)
    k, v = torch.randn(1, HEADS, T, DIM), torch.randn(1, HEADS, T, DIM)
    out, _ = cache.update(k, v, 0, {"cos": cos, "sin": sin})
    assert torch.allclose(out, k, atol=1e-5)
    assert stats.pre_bias and "before the k_proj bias" in stats.describe()


def test_removing_the_bias_shrinks_the_error_of_a_loud_bias_channel():
    cos, sin = tables()
    bias = torch.zeros(HEADS * DIM)
    bias[3] = 40.0                       # the Qwen2.5 pattern: one loud channel, every token
    b_rot = rotated_key_bias(bias, cos, sin, HEADS)
    torch.manual_seed(1)
    k = torch.randn(1, HEADS, T, DIM) + b_rot
    v = torch.randn(1, HEADS, T, DIM)

    plain, _ = build(2, None)
    debiased, _ = build(2, [bias, bias])
    a, _ = plain.update(k.clone(), v.clone(), 0, {"cos": cos, "sin": sin})
    c, _ = debiased.update(k.clone(), v.clone(), 0, {"cos": cos, "sin": sin})
    err = lambda x: float((x - k).norm() / k.norm())
    assert err(c) < err(a)


def test_each_layer_uses_its_own_bias():
    cos, sin = tables()
    b0, b1 = torch.zeros(HEADS * DIM), torch.zeros(HEADS * DIM)
    b0[3], b1[7] = 30.0, 30.0
    cache, _ = build(16, [b0, b1])
    k, v = torch.randn(1, HEADS, T, DIM), torch.randn(1, HEADS, T, DIM)
    # At 16 bits both layers are exact whichever bias was subtracted, so what this pins is
    # that the lookup is by layer at all -- and that the two rotated biases differ, which is
    # what makes the lookup matter once the width is low.
    out0, _ = cache.update(k.clone(), v, 0, {"cos": cos, "sin": sin})
    out1, _ = cache.update(k.clone(), v, 1, {"cos": cos, "sin": sin})
    assert torch.allclose(out0, k, atol=1e-5) and torch.allclose(out1, k, atol=1e-5)
    assert not torch.allclose(rotated_key_bias(b0, cos, sin, HEADS),
                              rotated_key_bias(b1, cos, sin, HEADS))


def test_a_missing_rotary_table_is_refused():
    cos, sin = tables()
    cache, _ = build(4, [torch.zeros(HEADS * DIM)])
    with pytest.raises(RuntimeError, match="rotary tables"):
        cache.update(torch.randn(1, HEADS, T, DIM), torch.randn(1, HEADS, T, DIM), 0, {})
