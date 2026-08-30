"""The prefix cache for dense LLaDA, checked against a stub of the real layout.

The one property everything else rests on: forwarding only ``[lo:]`` while the
prefix comes out of the store must produce exactly the logits a full pass would
have produced at those positions. If that fails, every number measured off this
path is measuring the harness.

The stub reproduces the two things about the checkpoint that make the windowed
forward correct at all -- rotary phase taken from the key/query lengths (so a
``layer_past`` of length ``lo`` puts the window at positions ``lo..``) and
unmasked bidirectional attention -- without needing the 8B weights.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from dllmquant.models.llada_local import (
    bidirectional_bias,
    blocks,
    logits_from_lo,
    refresh_prefix,
    run_blocks,
)


D_MODEL = 16
VOCAB = 32
N_LAYERS = 3


class StubBlock(nn.Module):
    """One block with the checkpoint's attention contract.

    ``attention`` there is a method taking ``(q, k, v, attention_bias,
    layer_past, use_cache)`` and returning ``(out, present)``; the block wraps
    it and returns ``(x, cache)``. What matters for the cache is reproduced
    exactly: rotary phase is a function of position derived from the key and
    query lengths, so it lands on absolute positions when a past is supplied.
    """

    def __init__(self, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.out = nn.Linear(D_MODEL, D_MODEL, bias=False)
        with torch.no_grad():
            self.qkv.weight.copy_(torch.randn(3 * D_MODEL, D_MODEL, generator=g) * 0.1)
            self.out.weight.copy_(torch.randn(D_MODEL, D_MODEL, generator=g) * 0.1)

    @staticmethod
    def _phase(x: torch.Tensor, start: int) -> torch.Tensor:
        pos = torch.arange(start, start + x.shape[-2], dtype=x.dtype,
                           device=x.device).unsqueeze(-1)
        return x + 0.01 * torch.cos(pos * 0.7 + torch.arange(
            x.shape[-1], dtype=x.dtype, device=x.device))

    def forward(self, x, attention_bias=None, layer_past=None, use_cache=False):
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        past_len = 0 if layer_past is None else layer_past[0].shape[-2]
        # The checkpoint reads its rotary offset off the lengths, not off an
        # explicit position id: keys start at 0, queries start where the past
        # ends. That is the whole reason a windowed forward lands correctly.
        k = self._phase(k, past_len)
        q = self._phase(q, past_len)

        present = (k, v) if use_cache else None
        if layer_past is not None:
            k = torch.cat([layer_past[0], k], dim=-2)
            v = torch.cat([layer_past[1], v], dim=-2)

        scores = q @ k.transpose(-1, -2) / (D_MODEL ** 0.5)
        if attention_bias is not None:
            # The checkpoint hands the bias down whole and each block cuts its
            # own view out of it. Reproduced exactly, because getting this
            # wrong is invisible without a past and fatal with one -- the
            # first version of this file passed a pre-sliced bias and the
            # mismatch surfaced inside SDPA naming neither the bias nor the
            # window.
            q_len, k_len = q.shape[-2], k.shape[-2]
            bias = attention_bias[:, :, k_len - q_len:k_len, :k_len]
            scores = scores + bias[0]
        att = torch.softmax(scores, dim=-1) @ v
        return x + self.out(att), present


class StubTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, D_MODEL)
        self.emb_drop = nn.Identity()
        self.blocks = nn.ModuleList(StubBlock(i) for i in range(N_LAYERS))
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.ff_out = nn.Linear(D_MODEL, VOCAB, bias=False)


class StubConfig:
    input_emb_norm = False
    d_model = D_MODEL


class StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = StubTransformer()
        self.config = StubConfig()


class StubWrapped(nn.Module):
    """The other layout: inner model behind an LM head at ``.model``."""

    def __init__(self, inner: StubModel):
        super().__init__()
        self.model = inner
        self.config = inner.config


@pytest.fixture()
def model():
    torch.manual_seed(0)
    m = StubModel().eval()
    return m


def test_finds_blocks_under_both_layouts(model):
    assert len(blocks(model)) == N_LAYERS
    assert len(blocks(StubWrapped(model))) == N_LAYERS


def test_bias_is_a_zero_square_over_all_keys():
    """Square, not window-shaped: the block cuts its own slice out of it."""
    bias = bidirectional_bias(10, torch.float32, "cpu")
    assert bias.shape == (1, 1, 10, 10)
    assert torch.count_nonzero(bias) == 0


def test_windowed_forward_matches_the_full_pass(model):
    """The load-bearing one: a cached prefix must change nothing."""
    x = torch.randint(0, VOCAB, (1, 12))
    lo = 5

    full, harvested = run_blocks(model, x, None, collect=True)
    past = [(k[..., :lo, :], v[..., :lo, :]) for k, v in harvested]
    windowed, _ = run_blocks(model, x[:, lo:], past)

    assert windowed.shape == full[:, lo:, :].shape
    torch.testing.assert_close(windowed, full[:, lo:, :], rtol=1e-5, atol=1e-5)


def test_a_lossless_store_reproduces_the_full_pass(model):
    """Same thing again, but through the real store at 16 bits.

    Separated from the test above on purpose: that one checks the windowing
    arithmetic, this one checks that nothing in the quantizer's bookkeeping
    disturbs it when it is asked to be lossless.
    """
    from dllmquant.cache import BlockKVCache, KVCacheConfig

    x = torch.randint(0, VOCAB, (1, 12))
    lo = 5
    cache = BlockKVCache(
        KVCacheConfig(enabled=True, decoded_bits=16, masked_bits=16,
                      policy="every_n", refresh_every=1),
        N_LAYERS,
    )

    full, _ = run_blocks(model, x, None)
    refresh_prefix(model, cache, x, lo, mask_id=VOCAB - 1)
    windowed = logits_from_lo(model, cache, x, lo)

    torch.testing.assert_close(windowed, full[:, lo:, :], rtol=1e-3, atol=1e-3)


def test_no_prefix_falls_back_to_a_full_pass(model):
    """At block zero there is nothing to cache and lo is the prompt end."""
    from dllmquant.cache import BlockKVCache, KVCacheConfig

    x = torch.randint(0, VOCAB, (1, 12))
    cache = BlockKVCache(KVCacheConfig(enabled=True), N_LAYERS)

    full, _ = run_blocks(model, x, None)
    got = logits_from_lo(model, cache, x, 0)
    torch.testing.assert_close(got, full, rtol=1e-5, atol=1e-5)
