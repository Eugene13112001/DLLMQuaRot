"""A low-rank correction on top of the quantized cache: `K ~ Q3(K) + A.B`.

Established on weights (LQER, CALDERA, ZeroQuant-V2) and untried on a
diffusion LM's cache. What is tested here is the part that does not need the
checkpoint, and it is the part that decides whether the rest is worth a night
on a shared node: the correction competes against the same bits spent flat,
and the planning arithmetic that made it look cheap counted numbers rather
than bits.
"""

from __future__ import annotations

import pytest
import torch

import importlib.util
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from dllmquant.cache import (
    BlockKVCache,
    KVCacheConfig,
    lowrank_bits,
    lowrank_residual,
    mixed_precision,
    mixed_precision_bits,
    quantize_kv,
)

_spec = importlib.util.spec_from_file_location(
    "check_lowrank",
    __import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "check_lowrank.py",
)
cl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cl)



def test_the_plans_ten_percent_is_ten_percent_of_numbers_not_of_bits():
    """The correction to the planning arithmetic, pinned.

    A rank-8 correction on a 224x128 prefix is 2816 numbers against 28672
    entries, which is the 10% the plan quotes. But the entries are three bits
    and the numbers are fp16, so in bits it is 1.57 per entry and "3 bits +
    rank 8" is 4.57 effective -- losing to four bits flat before it is
    measured. The plan's 3.3 appears only when the factors are quantized too.
    """
    assert lowrank_bits(8, 224, 128, factor_bits=16) == pytest.approx(1.571, abs=1e-3)
    assert 3 + 0.286 + lowrank_bits(8, 224, 128, 16) == pytest.approx(4.86, abs=0.01)

    # Four-bit factors, their own scales included, are what reaches the
    # advertised figure -- and how finely those scales are cut decides most of
    # it. One per token is T of them; at rank 1 that is five times the factor
    # they describe, and it is what the first survey was really pricing.
    assert 3 + 0.286 + lowrank_bits(8, 224, 128, 4) == pytest.approx(3.70, abs=0.01)
    assert 3 + 0.286 + lowrank_bits(8, 224, 128, 4, factor_scales="token") \
        == pytest.approx(3.94, abs=0.01)
    assert lowrank_bits(1, 224, 128, 4) == pytest.approx(0.051, abs=1e-3)
    assert lowrank_bits(1, 224, 128, 4, factor_scales="token") \
        == pytest.approx(0.300, abs=1e-3)
    assert lowrank_bits(0, 224, 128) == 0.0


def test_a_rank_one_residual_is_recovered_almost_exactly():
    torch.manual_seed(0)
    base = torch.randn(1, 2, 32, 16)
    spike = torch.randn(1, 2, 32, 1) @ torch.randn(1, 2, 1, 16)

    correction, captured = lowrank_residual(base + spike, base, rank=1)
    assert captured > 0.99
    assert float((correction - spike).pow(2).mean()) < 1e-8


def test_more_rank_recovers_more_and_never_less():
    torch.manual_seed(1)
    x = torch.randn(1, 2, 64, 32)
    q = quantize_kv(x, 3, 32, axis="token")

    captured = [lowrank_residual(x, q, r)[1] for r in (1, 2, 4, 8)]
    assert captured == sorted(captured)
    left = [
        float((x - q - lowrank_residual(x, q, r)[0]).pow(2).mean())
        for r in (1, 8)
    ]
    assert left[1] < left[0] < float((x - q).pow(2).mean())


def test_one_more_flat_bit_removes_more_error_than_a_rank_four_correction():
    """On error alone, before either is priced.

    Gaussian channels with the gains this model's outlier survey reports. The
    residual is genuinely low-rank -- per-channel scales make the wide channels
    round coarsest, so the error concentrates where they are -- and a whole
    extra bit still removes more of it than rank 4 does. Which of the two is
    the better *deal* depends on what each costs, and that is the survey's
    business, not this test's: with cheap factor scales the rank is several
    times cheaper than the bit.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 4, 224, 128)
    gain = torch.ones(128)
    gain[[7, 13, 61, 90]] = torch.tensor([4.0, 9.0, 6.0, 5.0])
    x = x * gain

    q = quantize_kv(x, 3, 128, axis="token")
    base = float((x - q).pow(2).mean().sqrt())

    correction, captured = lowrank_residual(x, q, rank=4)
    assert captured > 0.4, "the residual really does have a few fat directions"
    with_rank = float((x - q - correction).pow(2).mean().sqrt()) / base
    flat = float((x - quantize_kv(x, 4, 128, axis="token")).pow(2).mean().sqrt()) / base

    assert flat < with_rank, (flat, with_rank)
    # And it is cheaper, too: 0.79 bits for the rank against 1.00 for the bit
    # is the only thing the correction has going for it, and it is not enough.
    assert lowrank_bits(4, 224, 128) < 1.0


def test_the_cache_applies_the_correction_and_reports_what_it_recovered():
    torch.manual_seed(2)
    x = torch.randn(1, 2, 64, 32)
    x[..., 5] *= 8.0

    def err(rank):
        cache = BlockKVCache(
            KVCacheConfig(enabled=True, decoded_bits=3, masked_bits=3,
                          group_size=32, key_axis="token",
                          lowrank_rank=rank, lowrank_kinds=("key",)),
            n_layers=1,
        )
        kq, vq = cache.write(0, x, x)
        return float((kq - x).pow(2).mean()), float((vq - x).pow(2).mean()), cache

    plain_k, plain_v, plain = err(0)
    fixed_k, fixed_v, corrected = err(4)

    assert fixed_k < plain_k
    assert fixed_v == plain_v, "V was not asked for a correction"
    assert not plain.stats.lowrank_captured
    assert 0.0 < corrected.stats.lowrank_captured[0] < 1.0


def test_nothing_is_thrown_away_at_sixteen_bits_so_nothing_is_corrected():
    torch.manual_seed(3)
    x = torch.randn(1, 2, 16, 32)
    cache = BlockKVCache(
        KVCacheConfig(enabled=True, decoded_bits=16, masked_bits=16,
                      lowrank_rank=4),
        n_layers=1,
    )
    kq, _ = cache.write(0, x, x)
    assert torch.equal(kq, x)
    assert not cache.stats.lowrank_captured


def test_the_geometry_tells_a_sink_from_a_fat_channel():
    """A rank-1 residual is an outer product, and which side is concentrated
    decides which phase-C item this actually is.

    A single channel means precision spent there is the cheaper fix. A single
    position means an attention sink, and the correction is not a low-rank
    method at all -- it is the sink item wearing its clothes.
    """
    torch.manual_seed(0)
    plain = torch.randn(1, 4, 224, 128)
    fat_channel = plain.clone()
    fat_channel[..., 61] *= 20.0
    sink = plain.clone()
    sink[..., 3, :] *= 20.0

    spread = cl.residual_geometry([plain], 3, "token", 128)
    channel = cl.residual_geometry([fat_channel], 3, "token", 128)
    position = cl.residual_geometry([sink], 3, "token", 128)

    assert spread["tokens"] > 20 and spread["channels"] > 20
    assert channel["channels"] < 2 < channel["tokens"]
    assert position["tokens"] < 5 < position["channels"]
    assert position["modal_position"] == 3
    assert position["modal_agreement"] == 1.0


def test_widening_the_fat_channels_does_the_same_job_cheaper():
    """The control the geometry demanded, and the reason it is the right one.

    The residual's dominant direction is a single channel, and a rank-1 term is
    an outer product: it pays for a token factor as well, which encodes
    nothing the channel did not already say. Giving that channel more bits
    costs a fraction and leaves the same residual.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 4, 224, 128)
    gain = torch.ones(128)
    gain[[7, 13, 61, 90]] = torch.tensor([4.0, 9.0, 6.0, 5.0])
    x = x * gain

    q = quantize_kv(x, 3, 128, axis="token")
    base = float((x - q).pow(2).mean().sqrt())

    def left(t):
        return float((x - t).pow(2).mean().sqrt()) / base

    rank1 = left(q + lowrank_residual(x, q, 1, 4)[0])
    widened = left(mixed_precision(x, 3, 4, 1, 128, axis="token"))

    assert widened == pytest.approx(rank1, abs=0.02), "the same job"
    assert mixed_precision_bits(3, 4, 1, 128) - 3 < lowrank_bits(1, 224, 128, 4)

    # And it scales the same way: k channels tracks rank k.
    for k in (2, 4):
        assert left(mixed_precision(x, 3, 4, k, 128, axis="token")) == pytest.approx(
            left(q + lowrank_residual(x, q, k, 4)[0]), abs=0.02
        )


def test_the_index_list_is_the_only_thing_mixed_precision_adds():
    """Which channels are widest is a property of the layer, so the cache
    carries nothing extra: the same groups, wider values in a few of them."""
    assert mixed_precision_bits(3, 4, 1, 128) == pytest.approx(3.031, abs=1e-3)
    assert mixed_precision_bits(3, 4, 0, 128) == 3.0
    assert mixed_precision_bits(3, 8, 128, 128) == 11.0

    torch.manual_seed(1)
    x = torch.randn(1, 2, 16, 32)
    plain = quantize_kv(x, 3, 32, axis="token")
    assert torch.equal(mixed_precision(x, 3, 0, 4, 32), plain)
    assert torch.equal(mixed_precision(x, 3, 4, 0, 32), plain)
