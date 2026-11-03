"""Decoding loops against tiny fake models whose ideal output is known."""
from types import SimpleNamespace

import pytest
import torch
from transformers.cache_utils import DynamicCache

from blockkv import decode

V, MASK, STOP = 50, 49, 48


def target(pos):
    """The token the fake models want at absolute position `pos`."""
    return (pos * 7) % 40


def peaked(tokens, positions):
    """Logits with argmax `tokens`; positions divisible by 3 are low-confidence."""
    logits = torch.zeros(len(tokens), V)
    sharp = torch.where(positions % 3 == 0, 1.0, 12.0)
    logits[torch.arange(len(tokens)), tokens] = sharp
    return logits.unsqueeze(0)


def wanted(positions, stop_at):
    t = torch.tensor([target(int(p)) for p in positions])
    return torch.where(positions == stop_at, STOP, t)


class FakeSDAR:
    """Same-position prediction; stores KV only when store_kv=True."""

    def __init__(self, stop_at):
        self.stop_at = stop_at

    def __call__(self, x, attention_mask, position_ids, past_key_values, use_cache, store_kv):
        pos = position_ids[0]
        assert attention_mask.shape[-2:] == (len(pos), int(pos[-1]) + 1)
        assert past_key_values.get_seq_length() == int(pos[0])
        if store_kv:
            assert (x != MASK).all()
            kv = pos.float().view(1, 1, -1, 1)
            past_key_values.update(kv, kv, 0)
        return SimpleNamespace(logits=peaked(wanted(pos, self.stop_at), pos))


class FakeFastDLLM:
    """Logits at position i predict token i+1; positions follow the cache length."""

    def __init__(self, stop_at):
        self.stop_at = stop_at

    def __call__(self, input_ids, use_cache, past_key_values, update_past_key_values):
        cache = past_key_values if past_key_values is not None else DynamicCache()
        start = cache.get_seq_length()
        pos = torch.arange(start, start + input_ids.shape[1])
        if update_past_key_values:
            assert (input_ids != MASK).all()
            kv = pos.float().view(1, 1, -1, 1)
            cache.update(kv, kv, 0)
        return SimpleNamespace(logits=peaked(wanted(pos + 1, self.stop_at), pos + 1),
                               past_key_values=cache)


def check_stores(stores):
    """Stored ranges tile [0, end) with no gap or overlap."""
    edge = 0
    for lo, hi in stores:
        assert lo == edge
        edge = hi
    return edge


@pytest.mark.parametrize("block", [4, 8, 16])
@pytest.mark.parametrize("P", [9, 16, 23])
def test_sdar_reproduces_target(block, P):
    stop_at = P + 37
    stores = []
    gen, st = decode.sdar_generate(
        FakeSDAR(stop_at), torch.tensor([[target(p) for p in range(P)]]), MASK, block, block,
        0.9, 200, [STOP], on_store=lambda c, lo, hi: stores.append((lo, hi)))
    pos = torch.arange(P, stop_at + 1)
    assert torch.equal(gen, wanted(pos, stop_at))
    # Every generated block is stored after it finishes, and the current block never is.
    assert check_stores(stores) == (stop_at // block + 1) * block
    # Tokens committed in generated blocks = block slots minus the prompt tail in the first one.
    assert sum(st.per_step) == st.blocks * block - (P % block)
    # Low-confidence positions are committed one per step by the fallback.
    assert st.forced > 0 and st.steps < sum(st.per_step)
    assert st.nfe == st.steps + len(stores)


@pytest.mark.parametrize("small", [4, 8])
@pytest.mark.parametrize("P", [40, 64, 77])
def test_fastdllm_reproduces_target(small, P):
    block, stop_at = 32, P + 70
    stores = []
    gen, st = decode.fastdllm_generate(
        FakeFastDLLM(stop_at), torch.tensor([[target(p) for p in range(P)]]), MASK, block, small,
        1.0, 512, [STOP], on_store=lambda c, lo, hi: stores.append((lo, hi)))
    pos = torch.arange(P, stop_at + 1)
    assert torch.equal(gen, wanted(pos, stop_at))
    assert check_stores(stores) % block == 0
    # Threshold 1 commits exactly one token per step.
    assert set(st.per_step) == {1}
    assert st.forced == st.steps


def test_threshold_zero_commits_whole_block_in_one_step():
    P, block = 16, 8
    gen, st = decode.sdar_generate(FakeSDAR(P + 20), torch.tensor([[target(p) for p in range(P)]]),
                                   MASK, block, block, 0.0, 64, [STOP])
    assert set(st.per_step) == {block} and st.forced == 0
