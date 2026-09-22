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


def test_with_a_rotation_the_rotated_bias_is_what_leaves_the_store():
    # Under R4 the store receives RoPE(Wx + b) H. Taking out RoPE(b) without H is an
    # identity too, so it would pass an exactness test and still leave the bias in range.
    # The quantizer must see (k - RoPE(b) H): with a key that is nothing but the rotated
    # bias, that is zero, and a 2-bit store of zero plus the bias returns the key exactly.
    cos, sin = tables()
    bias = torch.zeros(HEADS * DIM)
    bias[3] = 40.0
    torch.manual_seed(3)
    h, _ = torch.linalg.qr(torch.randn(DIM, DIM))
    k = rotated_key_bias(bias, cos, sin, HEADS) @ h
    v = torch.randn(1, HEADS, T, DIM)
    stats = FDv2CacheStats(key_axis="channel", value_axis="channel")
    cls = make_quantized_cache_class(
        FakeCache, key_bits=2, value_bits=16, group_size=128, key_axis="channel",
        value_axis="channel", stats=stats, key_biases=[bias], kv_heads=HEADS,
        rotation=lambda: h)
    out, _ = cls().update(k.clone(), v, 0, {"cos": cos, "sin": sin})
    assert torch.allclose(out, k, atol=1e-4)
    # and without telling the cache about H, the same key is not recovered
    cls_wrong = make_quantized_cache_class(
        FakeCache, key_bits=2, value_bits=16, group_size=128, key_axis="channel",
        value_axis="channel", stats=FDv2CacheStats(), key_biases=[bias], kv_heads=HEADS)
    wrong, _ = cls_wrong().update(k.clone(), v, 0, {"cos": cos, "sin": sin})
    assert not torch.allclose(wrong, k, atol=1e-2)


def build_mean(bits, axis="channel"):
    stats = FDv2CacheStats(key_axis=axis, value_axis="channel")
    cls = make_quantized_cache_class(
        FakeCache, key_bits=bits, value_bits=16, group_size=128, key_axis=axis,
        value_axis="channel", stats=stats, key_mean=True)
    return cls()


def test_key_mean_is_exact_at_sixteen_bits():
    k, v = torch.randn(1, HEADS, T, DIM), torch.randn(1, HEADS, T, DIM)
    out, _ = build_mean(16).update(k.clone(), v, 0)
    assert torch.allclose(out, k, atol=1e-5)


def test_key_mean_catches_a_constant_offset_but_not_a_rotating_one():
    # A bias that does not rotate is one vector for every key and the mean takes it out as
    # well as pre-bias does. After RoPE it rotates with the position, and in a fast-rotating
    # pair the mean over positions is close to zero -- that part stays in the store.
    cos, sin = tables()
    bias = torch.zeros(HEADS * DIM)
    bias[0] = 40.0                         # pair (0, 8): the fastest rotation in the head
    b_rot = rotated_key_bias(bias, cos, sin, HEADS)
    torch.manual_seed(5)
    k = torch.randn(1, HEADS, T, DIM) + b_rot
    v = torch.randn(1, HEADS, T, DIM)
    err = lambda x: float((x - k).norm() / k.norm())
    exact = make_quantized_cache_class(
        FakeCache, key_bits=2, value_bits=16, group_size=128, key_axis="channel",
        value_axis="channel", stats=FDv2CacheStats(), key_biases=[bias], kv_heads=HEADS)()
    by_mean = build_mean(2)
    a, _ = exact.update(k.clone(), v, 0, {"cos": cos, "sin": sin})
    b, _ = by_mean.update(k.clone(), v, 0)
    assert err(a) < err(b)


def test_pre_bias_and_key_mean_are_not_combined():
    from dllmquant.models.fast_dllm_v2_local import install_quantized_cache

    class Dummy:
        pass

    with pytest.raises(ValueError, match="separate runs"):
        install_quantized_cache(Dummy(), bits=4, pre_bias=True, key_mean=True)
