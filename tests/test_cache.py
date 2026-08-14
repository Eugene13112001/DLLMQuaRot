"""Quantized KV cache: format, refresh policy, and error accounting."""

from __future__ import annotations

import pytest
import torch

from dllmquant.cache import BlockKVCache, KVCacheConfig, quantize_kv


# ------------------------------------------------------------------ format


def test_quantized_values_land_on_a_grid_per_group():
    x = torch.randn(1, 4, 10, 128)
    q = quantize_kv(x, bits=4, group_size=32)

    assert q.shape == x.shape
    # Each group of 32 channels has its own scale, so each holds at most 16
    # distinct values -- not 16 across the whole 128-wide row.
    for g in range(4):
        chunk = q[0, 0, 0, g * 32 : (g + 1) * 32]
        assert len(torch.unique(chunk)) <= 16


def test_grouping_beats_one_scale_per_token():
    """The finding that motivated the whole design, restated for the cache."""
    torch.manual_seed(0)
    x = torch.randn(1, 2, 16, 128)
    x[..., 61] *= 40.0  # one fat channel, as in a real key tensor

    wide = quantize_kv(x, bits=4, group_size=128)   # one scale per token/head
    fine = quantize_kv(x, bits=4, group_size=32)

    err_wide = float((wide - x).pow(2).mean())
    err_fine = float((fine - x).pow(2).mean())
    assert err_fine < err_wide


def test_more_bits_never_hurt():
    torch.manual_seed(1)
    x = torch.randn(1, 2, 8, 64)
    errs = [
        float((quantize_kv(x, bits=b, group_size=32) - x).pow(2).mean())
        for b in (2, 3, 4, 8)
    ]
    assert errs == sorted(errs, reverse=True), errs


def test_16_bits_is_a_no_op():
    x = torch.randn(1, 2, 4, 32)
    assert torch.equal(quantize_kv(x, bits=16, group_size=32), x)


def test_dtype_is_preserved():
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(1, 2, 4, 64, dtype=dtype)
        assert quantize_kv(x, bits=4, group_size=32).dtype == dtype


def test_indivisible_head_dim_is_refused():
    x = torch.randn(1, 1, 2, 100)
    with pytest.raises(ValueError):
        quantize_kv(x, bits=4, group_size=32)


def test_group_larger_than_head_dim_degrades_gracefully():
    """QuaRot's 128 on a 64-wide head means one scale per token per head."""
    x = torch.randn(1, 2, 4, 64)
    q = quantize_kv(x, bits=4, group_size=128)
    assert q.shape == x.shape
    assert len(torch.unique(q[0, 0, 0])) <= 16


# ------------------------------------------------------------ mask status


def test_masked_and_decoded_entries_can_carry_different_precision():
    """A masked position's K/V is overwritten as soon as its token commits;
    a decoded one is final. Precision can follow that."""
    torch.manual_seed(2)
    cache = BlockKVCache(
        KVCacheConfig(decoded_bits=8, masked_bits=2, group_size=32), n_layers=1
    )
    x = torch.randn(1, 2, 8, 64)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, :4] = True  # first half still masked

    out, _ = cache.write(0, x, x, mask)

    err_masked = float((out[:, :, :4] - x[:, :, :4]).pow(2).mean())
    err_decoded = float((out[:, :, 4:] - x[:, :, 4:]).pow(2).mean())
    assert err_masked > err_decoded * 5


def test_equal_bit_widths_ignore_the_mask():
    torch.manual_seed(3)
    cache = BlockKVCache(KVCacheConfig(decoded_bits=4, masked_bits=4), n_layers=1)
    x = torch.randn(1, 2, 8, 128)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, :4] = True

    with_mask, _ = cache.write(0, x, x, mask)
    cache.reset()
    without, _ = cache.write(0, x, x, None)
    assert torch.equal(with_mask, without)


# ---------------------------------------------------------- refresh policy


def test_first_access_always_refreshes():
    cache = BlockKVCache(KVCacheConfig(policy="never"), n_layers=2)
    assert cache.should_refresh(0, step=0) is True


def test_every_n_refreshes_on_schedule():
    cache = BlockKVCache(KVCacheConfig(policy="every_n", refresh_every=3), n_layers=1)
    x = torch.randn(1, 1, 4, 32)
    cache.write(0, x, x)

    for step in (1, 2):
        cache.step = step
        assert cache.should_refresh(0, step) is False
    cache.step = 3
    assert cache.should_refresh(0, 3) is True


def test_block_boundary_forces_a_refresh_under_every_policy():
    x = torch.randn(1, 1, 4, 32)
    for policy in ("never", "block", "every_n", "mask_ratio"):
        cache = BlockKVCache(KVCacheConfig(policy=policy), n_layers=1)
        cache.write(0, x, x)
        cache.step = 1
        assert cache.should_refresh(0, 1, block_boundary=True) is True, policy


def test_adaptive_interval_tracks_the_mask_ratio():
    """Everything moves early in the trajectory, almost nothing moves late."""
    cache = BlockKVCache(
        KVCacheConfig(policy="mask_ratio", min_interval=1, max_interval=16),
        n_layers=1,
    )
    assert cache.interval_for(1.0) == 1    # fully masked -> refresh every step
    assert cache.interval_for(0.0) == 16   # fully decoded -> rarely
    assert cache.interval_for(0.6) == 7     # 1 + 15*0.4
    assert cache.interval_for(0.5) == 8     # 8.5, rounded half-to-even

    intervals = [cache.interval_for(r) for r in (1.0, 0.75, 0.5, 0.25, 0.0)]
    assert intervals == sorted(intervals), intervals


def test_never_policy_holds_entries_until_the_block_ends():
    cache = BlockKVCache(KVCacheConfig(policy="never"), n_layers=1)
    x = torch.randn(1, 1, 4, 32)
    cache.write(0, x, x)
    for step in range(1, 50):
        cache.step = step
        assert cache.should_refresh(0, step) is False


def test_bad_policy_is_refused():
    with pytest.raises(ValueError):
        KVCacheConfig(policy="whenever-i-feel-like-it")


# --------------------------------------------------------- error accounting


def test_drift_separates_staleness_from_rounding():
    """The two error sources have to be measurable apart, or the refresh
    interval cannot be tuned against anything."""
    torch.manual_seed(4)
    cache = BlockKVCache(KVCacheConfig(decoded_bits=4, group_size=32), n_layers=1)

    k_old = torch.randn(1, 2, 8, 64)
    cache.write(0, k_old, k_old)
    cache.step = 5

    k_fresh = k_old + 0.3 * torch.randn_like(k_old)  # the sequence moved on
    d = cache.measure_drift(0, k_fresh, k_fresh)

    assert d["age"] == 5
    assert d["staleness"] > 0 and d["quantization"] > 0
    # A 0.3-sigma shift dwarfs 4-bit rounding.
    assert d["staleness"] > d["quantization"]


def test_no_drift_when_nothing_moved():
    torch.manual_seed(5)
    cache = BlockKVCache(KVCacheConfig(decoded_bits=4, group_size=32), n_layers=1)
    k = torch.randn(1, 2, 8, 64)
    cache.write(0, k, k)

    d = cache.measure_drift(0, k, k)
    assert d["staleness"] < 1e-6
    assert d["quantization"] > 0  # rounding remains


def test_stats_track_hits_and_ages():
    cache = BlockKVCache(KVCacheConfig(), n_layers=1)
    x = torch.randn(1, 1, 4, 128)
    cache.write(0, x, x)
    for step in (1, 2, 3):
        cache.step = step
        cache.read(0)

    assert cache.stats.refreshes == 1
    assert cache.stats.reuses == 3
    assert cache.stats.hit_rate == 0.75
    assert cache.stats.mean_age == 2.0
    assert "hit rate" in cache.stats.summary()


def test_reading_an_empty_layer_is_an_error():
    cache = BlockKVCache(KVCacheConfig(), n_layers=1)
    with pytest.raises(KeyError):
        cache.read(0)


# ------------------------------------------------------------ K/V asymmetry


def test_key_and_value_can_be_stored_at_different_widths():
    """K is consumed by the softmax, V is summed against fixed weights.

    Storing them identically is an assumption this codebase made everywhere
    and never tested; making it an option is what allows testing it.
    """
    cfg = KVCacheConfig(enabled=True, group_size=4, key_bits=16, value_bits=2)
    cache = BlockKVCache(cfg, n_layers=1)

    torch.manual_seed(0)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn(1, 2, 6, 4)
    kq, vq = cache.write(0, k, v)

    assert torch.equal(kq, k), "16 bits must store K untouched"
    assert not torch.allclose(vq, v), "2 bits must visibly damage V"


def test_the_split_defaults_to_the_old_behaviour():
    """Unset, both sides follow the status-based widths exactly as before."""
    plain = KVCacheConfig(enabled=True, group_size=4, decoded_bits=4, masked_bits=4)
    split = KVCacheConfig(enabled=True, group_size=4, decoded_bits=4, masked_bits=4,
                          key_bits=4, value_bits=4)

    torch.manual_seed(0)
    k, v = torch.randn(1, 2, 6, 4), torch.randn(1, 2, 6, 4)

    a = BlockKVCache(plain, n_layers=1).write(0, k.clone(), v.clone())
    b = BlockKVCache(split, n_layers=1).write(0, k.clone(), v.clone())

    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_the_two_splits_stay_independent():
    """Position status and K/V side answer different questions; overriding one
    must not silently answer the other."""
    cfg = KVCacheConfig(
        enabled=True, group_size=4,
        decoded_bits=8, masked_bits=2,   # status split
        key_bits=16,                     # K opts out of it entirely
    )
    cache = BlockKVCache(cfg, n_layers=1)

    torch.manual_seed(0)
    k, v = torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4)
    mask = torch.tensor([[True, True, False, False]])
    kq, vq = cache.write(0, k, v, mask=mask)

    assert torch.equal(kq, k), "key_bits=16 must override both status widths"
    # V still honours the status split: masked positions are coarser.
    masked_err = (vq[:, :, :2] - v[:, :, :2]).abs().mean()
    decoded_err = (vq[:, :, 2:] - v[:, :, 2:]).abs().mean()
    assert masked_err > decoded_err


def test_a_bad_width_is_still_refused():
    with pytest.raises(ValueError):
        KVCacheConfig(key_bits=1)
    with pytest.raises(ValueError):
        KVCacheConfig(value_bits=17)


# ------------------------------------------------------------ drift by depth


def _drifted(cache, layer, k_old, v_old, k_new, v_new):
    """Write one state, then measure against a different one."""
    cache.write(layer, k_old, v_old)
    cache.advance()
    return cache.measure_drift(layer, k_new, v_new)


def test_drift_is_recorded_against_the_layer_it_came_from():
    """The averages answer 'how wrong'; only the per-layer split answers
    'where', and a refresh interval is tuned against the second one."""
    cache = BlockKVCache(KVCacheConfig(enabled=True, group_size=4), n_layers=3)
    torch.manual_seed(0)

    for layer in (0, 2):
        k = torch.randn(1, 2, 4, 4)
        _drifted(cache, layer, k, k.clone(), k + 0.1 * (layer + 1), k.clone())

    by_layer = cache.stats.drift_by_layer()
    assert set(by_layer) == {0, 2}
    assert by_layer[0]["n"] == 1 and by_layer[2]["n"] == 1
    # Layer 2 was given the larger perturbation, so it must show more staleness.
    assert by_layer[2]["staleness"] > by_layer[0]["staleness"]


def test_the_two_error_sources_stay_separated_per_layer():
    cache = BlockKVCache(KVCacheConfig(enabled=True, decoded_bits=4,
                                       masked_bits=4, group_size=4), n_layers=2)
    torch.manual_seed(0)
    k = torch.randn(1, 2, 4, 4)

    # Same tensor back: no staleness at all, only what rounding did.
    out = _drifted(cache, 0, k, k.clone(), k.clone(), k.clone())
    assert out["staleness"] < out["quantization"]

    by_layer = cache.stats.drift_by_layer()
    assert by_layer[0]["quantization"] > 0


def test_the_summary_reports_depth_when_there_is_depth():
    cache = BlockKVCache(KVCacheConfig(enabled=True, group_size=4), n_layers=2)
    torch.manual_seed(0)
    for layer in (0, 1):
        k = torch.randn(1, 2, 4, 4)
        _drifted(cache, layer, k, k.clone(), k + 0.2, k.clone())

    text = cache.stats.summary()
    assert "by depth" in text


def test_scramble_destroys_information_but_not_the_distribution():
    """The chance floor has to be a floor, not a slightly worse cache."""
    cache = BlockKVCache(KVCacheConfig(enabled=True, decoded_bits=16,
                                       masked_bits=16), n_layers=2)
    torch.manual_seed(0)
    k, v = torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4)
    cache.write(0, k, v)
    cache.write(1, k.clone(), v.clone())
    before = {layer: cache._k[layer].clone() for layer in (0, 1)}

    cache.scramble(torch.Generator().manual_seed(0))

    for layer in (0, 1):
        after = cache._k[layer]
        assert not torch.equal(before[layer], after), "nothing moved"
        # Same rows, reordered: the marginal distribution is untouched, so the
        # floor is not confounded by a scale the model never sees.
        assert torch.equal(
            before[layer].sort(dim=-2).values, after.sort(dim=-2).values
        )

    # Independent draws per layer and per side. A shared permutation would be a
    # shuffled but coherent sequence -- a bag of words, which is information.
    assert not torch.equal(cache._k[0], cache._k[1])
    assert not torch.equal(
        cache._k[0].sort(dim=-2).indices, cache._v[0].sort(dim=-2).indices
    )
