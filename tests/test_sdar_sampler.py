"""SDAR's sampler with the committed store, on a toy model with SDAR's cache contract.

The toy model does what ``modeling_sdar.SDARAttention`` does with the cache and nothing
more: ``update`` on ``store_kv=True``, read ``cache[layer]`` and concatenate otherwise. So
these tests pin the hookup -- what gets written and when, that steps are counted, that 16
bits changes nothing and that a run repeats -- without the 8B checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dllmquant.committed_cache import (  # noqa: E402
    CommittedCacheConfig, CommittedCacheStats, check_writes, make_committed_cache_class,
)
from dllmquant.models.sdar_local import choose, sdar_generate, transfer_quota  # noqa: E402

V, MASK, STOP = 24, 23, 22


class Out:
    def __init__(self, logits):
        self.logits = logits


class ToyAttnModel(torch.nn.Module):
    """Two layers of one-head attention over the store plus the current tokens."""

    def __init__(self, d=8, layers=2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.randn(V, d, generator=g)
        self.wk = [torch.randn(d, d, generator=g) for _ in range(layers)]
        self.wv = [torch.randn(d, d, generator=g) for _ in range(layers)]
        self.head = torch.randn(d, V, generator=g) * 3
        self.layers = layers
        self.store_calls = []

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, ids, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=True, store_kv=False):
        h = self.emb[ids] + 0.1 * position_ids.unsqueeze(-1).float()
        if store_kv:
            self.store_calls.append(ids.shape[1])
        for i in range(self.layers):
            k = (h @ self.wk[i]).unsqueeze(1)            # [B, 1, T, d]
            v = (h @ self.wv[i]).unsqueeze(1)
            if store_kv:
                k, v = past_key_values.update(k, v, i)
            elif len(past_key_values) > i:
                pk, pv = past_key_values[i]
                k, v = torch.cat([pk, k], -2), torch.cat([pv, v], -2)
            att = torch.softmax(h.unsqueeze(1) @ k.transpose(-1, -2) / 3, -1)
            h = h + (att @ v).squeeze(1)
        logits = h @ self.head
        logits[..., MASK] = -1e9
        return Out(logits)


def run(bits=16, block=4, prompt_len=10, scale="per_channel", gen=24, seed=0):
    model = ToyAttnModel(seed=seed)
    cfg = CommittedCacheConfig(key_bits=bits, value_bits=bits, key_scale=scale)
    stats = CommittedCacheStats()
    cache = make_committed_cache_class(DynamicCache, cfg, stats)()
    prompt = torch.arange(prompt_len) % 20
    out, counts = sdar_generate(model, prompt, cache=cache, block_length=block,
                                gen_length=gen, threshold=0.9, mask_id=MASK,
                                stop_ids=(STOP,))
    return model, out, counts, stats


def test_quota_matches_the_script():
    assert transfer_quota(4, 4).tolist() == [1, 1, 1, 1]
    assert transfer_quota(8, 3).tolist() == [3, 3, 2]


def test_choose_takes_all_sure_tokens_or_falls_back():
    masked = torch.tensor([[True, True, True, False]])
    sure = torch.full((1, 4, V), -10.0)
    sure[0, :, 1] = 10.0
    _, take, forced = choose(sure, masked, 1, 0.9)
    assert take.tolist() == [[True, True, True, False]] and not forced
    flat = torch.zeros(1, 4, V)
    flat[0, 1, 2] = 1.0
    _, take, forced = choose(flat, masked, 1, 0.9)
    assert take.tolist() == [[False, True, False, False]] and forced


def test_store_is_written_once_per_block_and_never_mid_block():
    model, out, counts, stats = run(bits=2, block=4, prompt_len=10)
    assert model.store_calls[0] == 8                      # prompt, aligned down to blocks
    assert all(n == 4 for n in model.store_calls[1:])
    assert check_writes(stats, 4, expected_tokens=counts["stored_tokens"]) == []
    assert counts["stored_tokens"] == out.shape[1]
    assert counts["steps"] >= counts["blocks"]
    assert (out[0, :10] == torch.arange(10) % 20).all()   # the prompt is never rewritten
    assert not (out == MASK).any()


def test_sixteen_bits_matches_a_plain_cache():
    _, ours, c1, _ = run(bits=16)
    model = ToyAttnModel()
    plain, c2 = sdar_generate(model, torch.arange(10) % 20, cache=DynamicCache(),
                              block_length=4, gen_length=24, threshold=0.9,
                              mask_id=MASK, stop_ids=(STOP,))
    assert torch.equal(ours, plain) and c1 == c2


def test_a_run_repeats():
    _, a, ca, _ = run(bits=2, scale="per_token")
    _, b, cb, _ = run(bits=2, scale="per_token")
    assert torch.equal(a, b) and ca == cb


def test_two_bits_reaches_the_answer_through_the_store():
    """Rounding the store must be able to change the text: it is read, not bypassed."""
    changed = 0
    for seed in range(6):
        _, a, _, _ = run(bits=16, seed=seed, block=8, prompt_len=16)
        _, b, _, _ = run(bits=2, seed=seed, block=8, prompt_len=16, scale="per_token")
        changed += int(not torch.equal(a, b))
    assert changed > 0
