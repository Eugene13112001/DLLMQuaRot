import pytest
import torch
from transformers.cache_utils import DynamicCache

from blockkv import decode
from blockkv.kvquant import KVQuant, cache_layers, fake_quant

H, D, LAYERS = 2, 64, 3


MAXLEN = 512


def _full(layer):
    g = torch.Generator().manual_seed(layer)
    k = torch.randn(1, H, MAXLEN, D, generator=g) * torch.linspace(0.1, 5, D)
    v = torch.randn(1, H, MAXLEN, D, generator=g)
    return k.half(), v.half()


FULL = [_full(layer) for layer in range(LAYERS)]


def kv_at(lo, hi, layer):
    """Keys/values of positions [lo, hi): fixed per position, whatever the chunking."""
    k, v = FULL[layer]
    return k[:, :, lo:hi].clone(), v[:, :, lo:hi].clone()


def fill(cache, lo, hi):
    for layer in range(LAYERS):
        cache.update(*kv_at(lo, hi, layer), layer)


def stored(block, total, quant):
    """Write [0, total) block by block, calling the quantizer like decode does."""
    cache = DynamicCache()
    for lo in range(0, total, block):
        fill(cache, lo, lo + block)
        quant(cache, lo, lo + block)
    return cache


def distinct_per_slice(x, dim):
    x = x.movedim(dim, -1).reshape(-1, x.shape[dim])
    return max(len(torch.unique(row)) for row in x)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_fake_quant_levels_and_error(bits):
    x = torch.randn(64, 32).half()
    q = fake_quant(x, bits, 1)
    assert distinct_per_slice(q, 1) <= 2 ** bits
    step = (x.float().amax(1, keepdim=True) - x.float().amin(1, keepdim=True)) / (2 ** bits - 1)
    assert ((q.float() - x.float()).abs() <= step / 2 + 2e-2 * step + 1e-3).all()


def test_fake_quant_constant_slice_is_exact():
    x = torch.full((4, 32), 0.75).half()
    assert torch.equal(fake_quant(x, 2, 1), x)


@pytest.mark.parametrize("block", [4, 8, 16, 32, 64])
def test_variant_a_quantizes_whole_groups_only(block):
    quant = KVQuant(bits=2, group=32)
    cache = stored(block, 200 if block < 64 else 192, quant)
    end = cache.get_seq_length()
    assert quant.done == (end // 32) * 32
    for layer, (k, v) in enumerate(cache_layers(cache)):
        assert torch.equal(k[:, :, quant.done:], kv_at(quant.done, end, layer)[0])  # tail exact
        q_k, q_v = k[:, :, :quant.done], v[:, :, :quant.done]
        # Keys: per channel within each group of 32 tokens; values: per token within 32 channels.
        assert distinct_per_slice(q_k.reshape(1, H, -1, 32, D), 3) <= 4
        assert distinct_per_slice(q_v.reshape(1, H, -1, D // 32, 32), 4) <= 4


def test_same_cache_for_every_block_size():
    """Variant A: the granularity does not depend on the block, so neither does the result."""
    caches = [stored(b, 128, KVQuant(bits=2, group=32)) for b in (4, 16, 64)]
    for other in caches[1:]:
        for (k1, v1), (k2, v2) in zip(cache_layers(caches[0]), cache_layers(other)):
            assert torch.equal(k1, k2) and torch.equal(v1, v2)


def test_tail_window_and_sink_stay_exact():
    quant = KVQuant(bits=2, group=32, window=32, sink=4)
    cache = stored(8, 120, quant)
    # sink 4, then groups of 32 while at least 32 newest tokens remain: [4, 68).
    assert quant.done == 68
    for layer, (k, v) in enumerate(cache_layers(cache)):
        exact_k = torch.cat([kv_at(lo, lo + 8, layer)[0] for lo in range(0, 120, 8)], dim=2)
        assert torch.equal(k[:, :, :4], exact_k[:, :, :4])
        assert torch.equal(k[:, :, 68:], exact_k[:, :, 68:])
        assert not torch.equal(k[:, :, 4:68], exact_k[:, :, 4:68])


def test_summary_counts_honest_bits():
    quant = KVQuant(bits=2, group=32)
    stored(16, 96, quant)
    s = quant.summary(96)
    assert quant.honest_bits == 3 and s["kv_quantized"] == 96 and s["kv_bits_avg"] == 3
    assert 0 < s["kv_relerr_k"] < 1 and 0 < s["kv_relerr_v"] < 1


class FakeSDARWithKV:
    """Same-position prediction (as in test_decode) that stores real-shaped KV."""

    def __init__(self, stop_at):
        self.stop_at = stop_at

    def __call__(self, x, attention_mask, position_ids, past_key_values, use_cache, store_kv):
        from tests.test_decode import peaked, wanted
        pos = position_ids[0]
        if store_kv:
            lo, hi = int(pos[0]), int(pos[-1]) + 1
            fill(past_key_values, lo, hi)
        from types import SimpleNamespace
        return SimpleNamespace(logits=peaked(wanted(pos, self.stop_at), pos))


@pytest.mark.parametrize("block", [4, 16])
def test_sdar_loop_quantizes_committed_blocks(block):
    from tests.test_decode import target, MASK, STOP
    P, stop_at = 40, 40 + 90
    prompt = torch.tensor([[target(p) for p in range(P)]])
    quant = KVQuant(bits=2, group=32)
    gen, st = decode.sdar_generate(FakeSDARWithKV(stop_at), prompt, MASK, block, block, 0.9, 256,
                                   [STOP], on_store=quant)
    ref, _ = decode.sdar_generate(FakeSDARWithKV(stop_at), prompt, MASK, block, block, 0.9, 256, [STOP])
    assert torch.equal(gen, ref)   # the fake model ignores the cache: decoding is unchanged
    assert st.cache_len == (stop_at // block + 1) * block
    assert quant.done == (st.cache_len // 32) * 32


def test_16_bit_changes_nothing():
    """Self-check 1: the hook runs but must return the cache bit for bit."""
    quant = KVQuant(bits=16, group=32)
    cache = stored(8, 200, quant)
    assert quant.done == 192 and quant.honest_bits == 16
    for layer, (k, v) in enumerate(cache_layers(cache)):
        ref_k, ref_v = kv_at(0, 200, layer)
        assert torch.equal(k, ref_k) and torch.equal(v, ref_v)
    s = quant.summary(200)
    assert s["kv_bits_avg"] == 16 and s["kv_relerr_k"] == 0 and s["kv_relerr_v"] == 0


@pytest.mark.parametrize("only", ["k", "v"])
def test_keys_or_values_alone(only):
    quant = KVQuant(bits=2, group=32, only=only)
    cache = stored(16, 128, quant)
    both = stored(16, 128, KVQuant(bits=2, group=32))
    for layer, ((k, v), (bk, bv)) in enumerate(zip(cache_layers(cache), cache_layers(both))):
        ref_k, ref_v = kv_at(0, 128, layer)
        assert torch.equal(k, bk if only == "k" else ref_k)
        assert torch.equal(v, bv if only == "v" else ref_v)
    assert quant.honest_bits == (3 + 16) / 2
    s = quant.summary(128)
    assert (s["kv_relerr_k"] > 0) == (only == "k") and (s["kv_relerr_v"] > 0) == (only == "v")


def test_shared_scale_spans_the_batch():
    from blockkv.kvquant import quant_keys
    a = torch.randn(1, 2, 32, 64) * 0.1
    b = torch.randn(1, 2, 32, 64) * 10
    x = torch.cat([a, b]).half()
    alone = quant_keys(a.half(), 2, 32, "channel")
    assert torch.equal(quant_keys(x, 2, 32, "channel")[:1], alone)   # per sample: the neighbour is irrelevant
    shared = quant_keys(x, 2, 32, "channel", shared=True)[:1]
    assert (shared.float() - a).pow(2).sum() > 10 * (alone.float() - a).pow(2).sum()


def test_batches_follow_the_harness():
    from blockkv.fastdllm_batch import make_batches, pad_right
    assert make_batches([5, 1, 3, 2], 3) == [[1, 3, 2], [0]]
    ids, seq = pad_right([[7, 8, 9], [4]], 99, "cpu")
    assert ids.tolist() == [[7, 8, 9], [4, 99, 99]] and seq.tolist() == [3, 1]
