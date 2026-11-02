"""The committed-block store: rounds what is written, keeps what it is told to, counts cost.

These are the module's self-checks from the plan (stage 1.2), run on a real
``DynamicCache`` with a fake block loop standing in for the model: 16 bits is the original,
8 bits is close, the key error grows as the width shrinks, the current block never reaches
the store, and the token counts add up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dllmquant.committed_cache import (  # noqa: E402
    CommittedCacheConfig, CommittedCacheStats, check_writes, make_committed_cache_class,
)

H, D = 2, 128


def build(**kw):
    cfg = CommittedCacheConfig(**kw)
    stats = CommittedCacheStats()
    return make_committed_cache_class(DynamicCache, cfg, stats)(), cfg, stats


def kv(t, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, H, t, D, generator=g)
    k[..., 7] *= 15.0            # a fixed outlier channel, as in real keys
    return k, torch.randn(1, H, t, D, generator=g)


def run_blocks(cache, prompt, block, n_blocks, layers=2, seed=0):
    """Prompt, then ``n_blocks`` finished blocks, each written once per layer."""
    ks, vs = [], []
    for i, t in enumerate([prompt] + [block] * n_blocks):
        k, v = kv(t, seed + i)
        ks.append(k)
        vs.append(v)
        for layer in range(layers):
            ok, ov = cache.update(k, v, layer)
    return torch.cat(ks, -2), torch.cat(vs, -2), ok, ov


@pytest.mark.parametrize("scale", ["per_channel", "per_token"])
def test_sixteen_bits_stores_the_original(scale):
    cache, cfg, stats = build(key_bits=16, value_bits=16, key_scale=scale)
    k, v, ok, ov = run_blocks(cache, 20, 4, 10)
    assert torch.equal(ok, k) and torch.equal(ov, v)
    assert stats.honest_bits(cfg)["kv"] == 16.0


def rel(a, b):
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


@pytest.mark.parametrize("scale", ["per_channel", "per_token"])
def test_eight_bits_is_close(scale):
    cache, _, _ = build(key_bits=8, value_bits=8, key_scale=scale, residual="per_write")
    k, v, ok, ov = run_blocks(cache, 32, 32, 4)
    assert rel(ok, k) < 1e-3 and rel(ov, v) < 1e-3


@pytest.mark.parametrize("scale", ["per_channel", "per_token"])
def test_key_error_grows_as_width_shrinks(scale):
    errs = []
    for bits in (8, 4, 3, 2):
        cache, _, stats = build(key_bits=bits, value_bits=bits, key_scale=scale,
                                residual="per_write")
        run_blocks(cache, 32, 32, 4)
        errs.append(stats.key_rel_error())
    assert errs == sorted(errs) and errs[0] < errs[-1] / 100


def test_per_channel_beats_per_token_on_an_offset_channel():
    """KIVI's case: a channel carrying a large constant offset, as a k_proj bias does."""
    g = torch.Generator().manual_seed(0)
    k, v = torch.randn(1, H, 128, D, generator=g), torch.randn(1, H, 128, D, generator=g)
    k[..., 7] += 30.0
    out = {}
    for scale in ("per_channel", "per_token"):
        cache, _, stats = build(key_bits=2, value_bits=16, key_scale=scale,
                                residual="per_write")
        cache.update(k, v, 0)
        out[scale] = stats.k.err
    assert out["per_channel"] < out["per_token"] / 1.5


def test_buffer_keeps_the_tail_exact_until_a_group_fills():
    cache, cfg, stats = build(key_bits=2, value_bits=16, residual="buffer", group=32)
    k, _, ok, _ = run_blocks(cache, 30, 4, 3)        # 42 tokens: one full group + 10
    assert not torch.equal(ok[..., :32, :], k[..., :32, :])
    assert torch.equal(ok[..., 32:, :], k[..., 32:, :])
    assert stats.k.quantized == 32 * 2               # two layers
    # one more block: still 46 tokens, one group
    k2, v2 = kv(4, 99)
    for layer in range(2):
        ok, _ = cache.update(k2, v2, layer)
    assert torch.equal(ok[..., 32:, :], torch.cat([k, k2], -2)[..., 32:, :])
    # the rounded group is not rounded twice
    assert stats.k.quantized == 32 * 2


def test_buffer_rounds_aligned_groups_like_one_write():
    """Group by group, the buffer gives what rounding the whole run at once would."""
    a, _, _ = build(key_bits=2, value_bits=16, residual="buffer")
    b, _, _ = build(key_bits=2, value_bits=16, residual="per_write")
    k, v = kv(64)
    for i in range(16):
        oa, _ = a.update(k[..., 4 * i:4 * i + 4, :], v[..., 4 * i:4 * i + 4, :], 0)
    ob, _ = b.update(k, v, 0)
    assert torch.allclose(oa, ob)


@pytest.mark.parametrize("scale", ["per_channel", "per_token"])
def test_sink_position_stays_exact_and_out_of_the_range(scale):
    cache, _, _ = build(key_bits=2, value_bits=2, key_scale=scale, residual="per_write",
                        exact_positions=(2,))
    k, v = kv(32)
    k[..., 2, :] *= 50.0
    ok, ov = cache.update(k, v, 0)
    assert torch.equal(ok[..., 2, :], k[..., 2, :]) and torch.equal(ov[..., 2, :], v[..., 2, :])
    plain, _, _ = build(key_bits=2, value_bits=2, key_scale=scale, residual="per_write")
    pk, _ = plain.update(k, v, 0)
    others = [i for i in range(32) if i != 2]
    with_sink = rel(ok[..., others, :], k[..., others, :])
    without = rel(pk[..., others, :], k[..., others, :])
    if scale == "per_channel":      # the sink no longer sets its neighbours' range
        assert with_sink < without / 4
    else:                           # per-token groups never mixed rows to begin with
        assert with_sink == without


def test_sink_in_a_later_write_is_found_by_absolute_position():
    cache, _, _ = build(key_bits=2, value_bits=2, key_scale="per_token",
                        exact_positions=(21,))
    k, v, ok, ov = run_blocks(cache, 20, 4, 2)
    assert torch.equal(ok[..., 21, :], k[..., 21, :])
    assert not torch.equal(ok[..., 20, :], k[..., 20, :])


def test_honest_bits():
    # per-token at 2 bits, group 32 over 128 channels: 2 + 2*16/32 = 3 bits
    cache, cfg, stats = build(key_bits=2, value_bits=2, key_scale="per_token")
    run_blocks(cache, 32, 4, 8)
    hb = stats.honest_bits(cfg)
    assert hb["k"] == pytest.approx(3.0) and hb["v"] == pytest.approx(3.0)
    # per-channel per write at block 4: a scale and zero for every 4 tokens -> 2 + 32/4
    cache, cfg, stats = build(key_bits=2, value_bits=16, residual="per_write")
    run_blocks(cache, 32, 4, 8)
    n_tok = 32 + 32
    pairs_per_ch = 1 + 8
    assert stats.honest_bits(cfg)["k"] == pytest.approx(
        (n_tok * 2 + pairs_per_ch * 32) / n_tok)
    # buffer at block 4: 64 tokens, both groups full -> 3 bits; plus 4 more stay exact
    cache, cfg, stats = build(key_bits=2, value_bits=16, residual="buffer")
    run_blocks(cache, 32, 4, 9)
    assert stats.honest_bits(cfg)["k"] == pytest.approx((64 * 3 + 4 * 16) / 68)


def test_sink_costs_sixteen_bits():
    cache, cfg, stats = build(key_bits=2, value_bits=2, key_scale="per_token",
                              exact_positions=(2,))
    run_blocks(cache, 32, 32, 0, layers=1)
    assert stats.honest_bits(cfg)["k"] == pytest.approx((31 * 3 + 16) / 32)


def test_clean_block_loop_passes_the_write_check():
    cache, _, stats = build(key_bits=2, value_bits=2)
    run_blocks(cache, 37, 16, 5)
    assert check_writes(stats, 16, expected_tokens=37 + 80) == []


def test_a_model_that_writes_its_current_block_is_caught():
    """Write the block at every denoising step and crop it back: the check must object."""
    cache, _, stats = build(key_bits=2, value_bits=2)
    k, v = kv(20)
    cache.update(k, v, 0)
    for _ in range(3):
        for _step in range(4):
            kb, vb = kv(8, 5)
            cache.update(kb, vb, 0)
            cache.crop(cache.get_seq_length(0) - 8)
        cache.update(kb, vb, 0)
    problems = check_writes(stats, 8)
    assert any("crops" in p for p in problems)


def test_wrong_token_count_is_caught():
    cache, _, stats = build(key_bits=2, value_bits=2)
    run_blocks(cache, 20, 8, 3)
    assert check_writes(stats, 8, expected_tokens=20 + 8 * 4)


def test_bad_config_is_refused():
    with pytest.raises(ValueError):
        CommittedCacheConfig(key_scale="channel")
    with pytest.raises(ValueError):
        CommittedCacheConfig(residual="keep")
