"""The prefix cache has to be exact before it is allowed to be lossy.

Block-causal attention makes the K/V of earlier blocks constant while the
current block is decoded, so caching them changes nothing about what the model
computes -- and that is checkable directly, against a full forward, without
generating a single token or scoring a single answer. Only once that holds
does a difference in output mean what we want it to mean: the cost of storing
those K/V in four bits.

Nothing here imports the vendored modelling code: it needs transformers >=
4.56, while the LLaDA-1.5 half of this project runs on 4.46. The model-shaped
parts are stood in for; the rotary and attention kernels are injected, and in
production they come from the checkpoint's own file so there is no second
implementation to keep in step.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from dllmquant.cache import BlockKVCache, KVCacheConfig
from dllmquant.models.llada2_local import (
    block_causal_mask,
    forward_window,
    install_block_cache,
    logits_for_window,
    refresh_prefix,
    remove_block_cache,
)

D, HEADS, KV_HEADS, HEAD_DIM, VOCAB = 32, 4, 2, 8, 40
BLOCK = 4


def _rotary(q, k, cos, sin):
    """Position-independent stand-in: the cache logic does not depend on RoPE,
    and the real one comes from the vendored file."""
    return q, k


def _attention(module, query, key, value, attention_mask, dropout=0.0, scaling=1.0, **kw):
    key = key.repeat_interleave(module.num_key_value_groups, dim=1)
    value = value.repeat_interleave(module.num_key_value_groups, dim=1)
    weights = (query @ key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask[:, :, :, : key.shape[-2]]
    weights = weights.softmax(dim=-1)
    return (weights @ value).transpose(1, 2).contiguous(), weights


class _Attention(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        self.query_key_value = nn.Linear(D, (HEADS + 2 * KV_HEADS) * HEAD_DIM, bias=False)
        self.dense = nn.Linear(HEADS * HEAD_DIM, D, bias=False)
        self.num_heads = HEADS
        self.num_key_value_heads = KV_HEADS
        self.num_key_value_groups = HEADS // KV_HEADS
        self.head_dim = HEAD_DIM
        self.scaling = HEAD_DIM**-0.5
        self.layer_idx = layer_idx
        self.config = SimpleNamespace(use_qk_norm=False)


class _Layer(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        self.attention = _Attention(layer_idx)
        self.mlp = nn.Linear(D, D, bias=False)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                position_embeddings=None, **kw):
        attn, _, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + attn
        return (hidden_states + self.mlp(hidden_states),)


class _Inner(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.word_embeddings = nn.Embedding(VOCAB, D)
        self.layers = nn.ModuleList([_Layer(i) for i in range(n_layers)])
        self.norm = nn.LayerNorm(D)

    def rotary_emb(self, hidden, position_ids):
        zeros = torch.zeros(1, position_ids.shape[-1], HEAD_DIM)
        return zeros, zeros


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()
        self.lm_head = nn.Linear(D, VOCAB, bias=False)


def _setup(bits: int = 16):
    torch.manual_seed(0)
    model = _Model().eval()
    cache = BlockKVCache(
        KVCacheConfig(enabled=True, decoded_bits=bits, masked_bits=bits,
                      group_size=HEAD_DIM),
        n_layers=len(model.model.layers),
    )
    states = install_block_cache(model, cache, rotary_fn=_rotary, attention_fn=_attention)
    return model, cache, states


def _full_logits(model, x):
    mask = block_causal_mask(x.shape[-1], BLOCK, dtype=torch.float32)
    positions = torch.arange(x.shape[-1]).unsqueeze(0)
    return forward_window(model, x, positions, mask)


# ------------------------------------------------------------------ masks


def test_a_window_mask_is_the_rows_of_the_square_one():
    square = block_causal_mask(12, BLOCK, dtype=torch.float32)
    rows = block_causal_mask(12, BLOCK, rows=(4, 8), dtype=torch.float32)

    assert rows.shape == (1, 1, 4, 12)
    assert torch.equal(rows[0, 0], square[0, 0, 4:8, :])


def test_a_window_sees_its_own_block_and_every_earlier_one():
    rows = block_causal_mask(12, BLOCK, rows=(4, 8), dtype=torch.float32)[0, 0]
    assert (rows[:, :8] == 0).all()          # earlier block and its own
    assert torch.isinf(rows[:, 8:]).all()    # not the next one


# --------------------------------------------------------------- exactness


def test_the_cached_window_matches_a_full_forward():
    """The claim the whole design rests on: reusing the prefix is not an
    approximation, because block-causal attention makes it constant."""
    model, _, states = _setup(bits=16)
    x = torch.randint(0, VOCAB, (1, 12))

    full = _full_logits(model, x)
    refresh_prefix(model, states, x, prefix_len=8, block_length=BLOCK)
    windowed = logits_for_window(model, states, x, lo=8, hi=12, block_length=BLOCK)

    assert windowed.shape == (1, 4, VOCAB)
    assert torch.allclose(windowed, full[:, 8:12], atol=1e-5, rtol=1e-4)


def test_changing_the_current_block_does_not_disturb_the_cache():
    """Tokens are committed inside the block between steps; the prefix must
    stay valid across all of them, which is what makes one refresh per block
    enough."""
    model, _, states = _setup(bits=16)
    x = torch.randint(0, VOCAB, (1, 12))
    refresh_prefix(model, states, x, prefix_len=8, block_length=BLOCK)

    for _ in range(3):
        x[:, 8:] = torch.randint(0, VOCAB, (1, 4))
        windowed = logits_for_window(model, states, x, lo=8, hi=12, block_length=BLOCK)
        full = _full_logits(model, x)
        assert torch.allclose(windowed, full[:, 8:12], atol=1e-5, rtol=1e-4)


def test_a_stale_prefix_is_detectably_wrong():
    """Guards the test above from passing for the wrong reason -- if the
    windowed path ignored the cache, everything would agree anyway."""
    model, _, states = _setup(bits=16)
    x = torch.randint(0, VOCAB, (1, 12))
    refresh_prefix(model, states, x, prefix_len=8, block_length=BLOCK)

    x[:, :8] = torch.randint(0, VOCAB, (1, 8))  # rewrite the prefix, do not refresh
    windowed = logits_for_window(model, states, x, lo=8, hi=12, block_length=BLOCK)

    assert not torch.allclose(windowed, _full_logits(model, x)[:, 8:12], atol=1e-3)


# ------------------------------------------------------------ quantization


def test_four_bit_storage_is_the_only_difference_left():
    model, _, states = _setup(bits=4)
    x = torch.randint(0, VOCAB, (1, 12))

    full = _full_logits(model, x)
    refresh_prefix(model, states, x, prefix_len=8, block_length=BLOCK)
    windowed = logits_for_window(model, states, x, lo=8, hi=12, block_length=BLOCK)

    assert torch.isfinite(windowed).all()
    assert not torch.allclose(windowed, full[:, 8:12], atol=1e-5)
    # Still recognisably the same function, not noise.
    error = (windowed - full[:, 8:12]).abs().mean() / full[:, 8:12].abs().mean()
    assert error < 0.5


def test_the_cache_records_what_it_was_asked_for():
    model, cache, states = _setup(bits=16)
    x = torch.randint(0, VOCAB, (1, 12))
    refresh_prefix(model, states, x, prefix_len=8, block_length=BLOCK)

    assert cache.has(0) and cache.has(1)
    key, value = cache.read(0)
    assert key.shape == (1, KV_HEADS, 8, HEAD_DIM)
    assert value.shape == (1, KV_HEADS, 8, HEAD_DIM)


# ------------------------------------------------------------- installation


def test_installing_leaves_every_module_name_untouched():
    """A wrapper module would rename layers.N.attention.query_key_value, and
    the rotation plan, the skip lists and the IA-AQ probe all read that name."""
    torch.manual_seed(0)
    model = _Model()
    before = [name for name, _ in model.named_modules()]

    _setup(bits=16)
    install_block_cache(
        model, BlockKVCache(KVCacheConfig(), n_layers=2),
        rotary_fn=_rotary, attention_fn=_attention,
    )

    assert [name for name, _ in model.named_modules()] == before


def test_removing_the_cache_puts_the_original_forward_back():
    model, _, _ = _setup(bits=16)
    assert "forward" in model.model.layers[0].attention.__dict__

    remove_block_cache(model)
    assert "forward" not in model.model.layers[0].attention.__dict__
    assert "_dllm_cache_state" not in model.model.layers[0].attention.__dict__


def test_an_uninitialised_cache_is_refused_rather_than_guessed():
    model, _, states = _setup(bits=16)
    x = torch.randint(0, VOCAB, (1, 12))

    with pytest.raises(KeyError):
        logits_for_window(model, states, x, lo=8, hi=12, block_length=BLOCK)


# ------------------------------------------------------------ cached sampler


def _sampler_setup(bits: int):
    """A model, a cache, and an adapter shaped like the LLaDA2.0 one."""
    from dllmquant.config import TMASConfig

    torch.manual_seed(0)
    model = _Model().eval()
    cache = BlockKVCache(
        KVCacheConfig(enabled=True, decoded_bits=bits, masked_bits=bits,
                      group_size=HEAD_DIM),
        n_layers=len(model.model.layers),
    )
    # The stand-in attention has no forward of its own -- the cache-aware one
    # is what gives it one. Install it now so the dense comparison has a model
    # that runs; cached_generate installs its own and removes it afterwards.
    install_block_cache(model, cache, rotary_fn=_rotary, attention_fn=_attention)

    class _Adapter:
        mask_id = VOCAB - 1

        def __init__(self):
            self.model = model
            self.block_length = BLOCK

        def _sequence_layout(self, p_len, cfg):
            b = cfg.block_length
            self.block_length = b
            n_blocks = (p_len + cfg.gen_length + b - 1) // b
            first = p_len // b
            return n_blocks * b, [(i * b, (i + 1) * b) for i in range(first, n_blocks)]

    return _Adapter(), cache, TMASConfig(gen_length=8, block_length=BLOCK, steps=4)


def _dense_generate(adapter, prompt, cfg):
    """The same sampler with no cache: recompute everything, every step."""
    from dllmquant.models.base import _split_evenly, get_num_transfer_tokens

    p_len = prompt.shape[-1]
    total, bounds = adapter._sequence_layout(p_len, cfg)
    x = torch.full((1, total), adapter.mask_id, dtype=torch.long)
    x[:, :p_len] = prompt
    schedule = _split_evenly(cfg.steps, len(bounds))

    for bi, (lo, hi) in enumerate(bounds):
        budget = get_num_transfer_tokens(x[:, lo:hi] == adapter.mask_id, schedule[bi])
        for step in range(schedule[bi]):
            logits = _full_logits(adapter.model, x)[:, lo:hi]
            conf, proposal = torch.softmax(logits.float(), -1).max(-1)
            masked = x[:, lo:hi] == adapter.mask_id
            k = int(budget[0, step])
            if k <= 0 or not masked.any():
                continue
            score = torch.where(masked, conf, torch.full_like(conf, -torch.inf))
            chosen = torch.topk(score[0], k=min(k, int(masked.sum()))).indices
            x[0, lo + chosen] = proposal[0, chosen]
    return x


def test_a_sixteen_bit_cache_commits_exactly_the_dense_tokens():
    """The claim the cached sampler rests on, as a test rather than an
    argument: refreshing the prefix once per block is not an approximation,
    because block-causal attention makes it constant for that block's whole
    decoding."""
    from dllmquant.models.llada2_local import cached_generate

    adapter, cache, cfg = _sampler_setup(bits=16)
    prompt = torch.tensor([1, 2, 3, 4, 5, 6])

    dense = _dense_generate(adapter, prompt, cfg)
    cached = cached_generate(adapter, prompt, cfg, cache,
                             rotary_fn=_rotary, attention_fn=_attention)

    assert torch.equal(cached, dense)


def test_four_bit_storage_is_allowed_to_change_the_answer():
    """Guards the test above from passing because the cache is ignored."""
    from dllmquant.models.llada2_local import cached_generate

    adapter, cache, cfg = _sampler_setup(bits=4)
    prompt = torch.tensor([1, 2, 3, 4, 5, 6])

    dense = _dense_generate(adapter, prompt, cfg)
    cached = cached_generate(adapter, prompt, cfg, cache,
                             rotary_fn=_rotary, attention_fn=_attention)

    assert cached.shape == dense.shape
    assert (cached[0, prompt.shape[-1]:] != adapter.mask_id).all()


def test_the_model_is_left_without_hooks_afterwards():
    """Hooks that survive the call would make the next question slower and
    quietly keep a cache from the previous one alive."""
    from dllmquant.models.llada2_local import cached_generate

    adapter, cache, cfg = _sampler_setup(bits=16)
    cached_generate(adapter, torch.tensor([1, 2, 3, 4]), cfg, cache,
                    rotary_fn=_rotary, attention_fn=_attention)

    for layer in adapter.model.model.layers:
        assert "forward" not in layer.attention.__dict__


def test_masked_and_decoded_positions_can_be_stored_at_different_widths():
    """A masked position's K/V is overwritten the moment its token is
    committed; a decoded one's is final. The cache can spend accordingly --
    but only because the writer is told which positions are which, and that
    only happens if the sampler passes the mask id down."""
    from dllmquant.models.llada2_local import cached_generate

    adapter, _, cfg = _sampler_setup(bits=16)
    cache = BlockKVCache(
        KVCacheConfig(enabled=True, decoded_bits=16, masked_bits=2,
                      group_size=HEAD_DIM),
        n_layers=len(adapter.model.model.layers),
    )
    prompt = torch.tensor([1, 2, 3, 4, 5, 6])

    cached_generate(adapter, prompt, cfg, cache,
                    rotary_fn=_rotary, attention_fn=_attention)

    assert cache.has(0), "nothing was ever written"


def test_without_a_mask_the_split_cannot_apply():
    """Guards the plumbing above: refresh_prefix called with no mask id has to
    fall back to one width rather than silently pick the masked one."""
    model, _, states = _setup(bits=16)
    cache = BlockKVCache(
        KVCacheConfig(enabled=True, decoded_bits=16, masked_bits=2,
                      group_size=HEAD_DIM),
        n_layers=len(model.model.layers),
    )
    for state in states:
        state.cache = cache

    x = torch.randint(0, VOCAB - 1, (1, 12))
    refresh_prefix(model, states, x, prefix_len=8, block_length=BLOCK)

    k, _ = cache.read(0)
    # decoded_bits is 16, so with no mask the tensor comes back untouched.
    assert torch.isfinite(k).all()
    assert len(k.unique()) > 2**2, "the 2-bit masked width was applied to all"


def test_the_mean_logit_error_carries_a_standard_error():
    """It is the metric two of this project's conclusions rest on.

    `argmax@k` is a rate over a few dozen committed positions and cannot
    resolve a few percent; `rel` is an average over whole logit tensors and
    can -- but only once its spread across canvases is carried alongside it.
    Without that, one row reading +0.5% and the next -3.4% is unreadable.
    """
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "check_block_cache",
        Path(__file__).resolve().parents[1] / "scripts" / "check_block_cache.py",
    )
    cbc = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass looks its class's module up in
    # sys.modules and fails on None if it is not there.
    sys.modules[spec.name] = cbc
    spec.loader.exec_module(cbc)

    torch.manual_seed(0)
    pooled = cbc.Comparison()
    errors = []
    for _ in range(16):
        ref = torch.randn(1, 4, 32)
        act = ref + 0.1 * torch.randn_like(ref)
        c = cbc.compare(ref, act)
        errors.append(c.rel)
        pooled = pooled + c

    n = len(errors)
    mean = sum(errors) / n
    var = sum((e - mean) ** 2 for e in errors) / (n - 1)

    assert pooled.rel == pytest.approx(mean, rel=1e-6)
    assert pooled.rel_se == pytest.approx((var / n) ** 0.5, rel=1e-6)

    # One canvas has a mean and no spread, and must say so rather than zero.
    single = cbc.compare(torch.randn(1, 4, 32), torch.randn(1, 4, 32))
    assert single.rel_se != single.rel_se


# ------------------------------------------------- reusing the current block


from dllmquant.models.llada2_local import cached_generate  # noqa: E402


def test_reusing_the_current_block_is_off_by_default():
    """Every number this project has taken was taken with it off, so the flag
    must default to the path that produced them and change nothing."""
    adapter, cache, cfg = _sampler_setup(bits=16)
    prompt = torch.randint(0, VOCAB - 1, (1, BLOCK))

    # Dense first: cached_generate removes the attention forward on its way
    # out, and the dense path needs it.
    dense = _dense_generate(adapter, prompt, cfg)
    out = cached_generate(adapter, prompt, cfg, cache,
                          rotary_fn=_rotary, attention_fn=_attention)

    assert cache.stats.window_refreshes == 0
    assert cache.stats.window_reuses == 0
    assert torch.equal(out, dense)


def test_a_reused_block_is_stored_once_and_read_after():
    adapter, cache, cfg = _sampler_setup(bits=16)
    cache.cfg.policy = "block"          # never refresh inside a block
    prompt = torch.randint(0, VOCAB - 1, (1, BLOCK))

    cached_generate(adapter, prompt, cfg, cache, reuse_window=True,
                    rotary_fn=_rotary, attention_fn=_attention)

    n_layers = len(adapter.model.model.layers)
    # One write per layer per block, and every later step reads.
    assert cache.stats.window_refreshes >= n_layers
    assert cache.stats.window_reuses > 0
    assert 0.0 < cache.stats.window_hit_rate < 1.0
    assert cache.stats.mean_window_age > 0


def test_a_reused_block_is_wrong_the_moment_a_token_commits():
    """Staleness, isolated from rounding at 16 bits, measured on the logits.

    Not on the sampler's output: the stand-in model commits the same token
    everywhere, so a decision has nowhere to move and the test would pass
    while measuring nothing. The logits are where the effect is, and this is
    the only configuration in the codebase where the cache can be wrong for a
    reason other than its bit width.
    """
    model, cache, states = _setup(bits=16)
    x = torch.randint(0, VOCAB - 1, (1, 12))
    prefix_len = 8

    refresh_prefix(model, states, x, prefix_len, BLOCK)
    logits_for_window(model, states, x, prefix_len, 12, BLOCK, window="record")

    # A token in the current block commits: everything stored above is now
    # keyed on a canvas that no longer exists.
    moved = x.clone()
    moved[0, prefix_len + 1] = (int(x[0, prefix_len + 1]) + 1) % (VOCAB - 1)

    fresh = logits_for_window(model, states, moved, prefix_len, 12, BLOCK)
    stale = logits_for_window(model, states, moved, prefix_len, 12, BLOCK,
                              window="reuse")

    full = _full_logits(model, moved)[:, prefix_len:12]
    assert torch.allclose(fresh, full, atol=1e-5, rtol=1e-4), "the exact path moved"
    assert not torch.allclose(stale, fresh, atol=1e-4), "reuse cost nothing"
    assert cache.stats.window_reuses == len(model.model.layers)


def test_staleness_grows_with_what_has_been_committed_since():
    model, cache, states = _setup(bits=16)
    x = torch.randint(0, VOCAB - 1, (1, 12))
    prefix_len = 8

    refresh_prefix(model, states, x, prefix_len, BLOCK)
    logits_for_window(model, states, x, prefix_len, 12, BLOCK, window="record")

    def drift(n_committed):
        moved = x.clone()
        for i in range(n_committed):
            moved[0, prefix_len + i] = (int(x[0, prefix_len + i]) + 1) % (VOCAB - 1)
        fresh = logits_for_window(model, states, moved, prefix_len, 12, BLOCK)
        stale = logits_for_window(model, states, moved, prefix_len, 12, BLOCK,
                                  window="reuse")
        return float((stale - fresh).abs().mean())

    one, three = drift(1), drift(3)
    assert 0.0 < one < three, (one, three)


def test_the_block_store_does_not_survive_the_block():
    """Indexed by layer, not by position: carrying it over would hand the next
    block the previous one's keys at the previous one's positions."""
    cache = BlockKVCache(KVCacheConfig(enabled=True, decoded_bits=16,
                                       masked_bits=16), n_layers=1)
    k = torch.randn(1, 2, 4, 8)
    cache.write_window(0, k, k)
    assert cache.has_window(0)

    cache.reset_window()
    assert not cache.has_window(0)
    with pytest.raises(KeyError):
        cache.read_window(0)


def test_the_two_stores_answer_to_different_clocks():
    """The prefix is rewritten once per block whatever the policy says; the
    current block ages every step. One interval for both would tie the cheap
    decision to the expensive one."""
    cache = BlockKVCache(
        KVCacheConfig(enabled=True, policy="every_n", refresh_every=3), n_layers=1
    )
    k = torch.randn(1, 2, 4, 128)
    cache.write(0, k, k)
    cache.write_window(0, k, k)

    for step in (1, 2):
        cache.step = step
        assert cache.should_refresh_window(0, step) is False
    cache.step = 3
    assert cache.should_refresh_window(0, 3) is True

    # And under 'block' the current block is never refreshed inside it, while
    # the prefix still is at the boundary.
    cache.cfg.policy = "block"
    assert cache.should_refresh_window(0, 99) is False
    assert cache.should_refresh(0, 99, block_boundary=True) is True


def test_an_empty_block_store_asks_to_be_filled():
    cache = BlockKVCache(KVCacheConfig(enabled=True, policy="block"), n_layers=1)
    assert cache.should_refresh_window(0, step=0) is True


def test_refreshing_every_step_makes_reuse_a_no_op():
    """The identity that validates the whole reuse path.

    With the block refreshed at every step there is nothing stale to read, so
    turning reuse on must change nothing at all. If this fails, any damage the
    trajectory sweep reports is the wiring rather than staleness -- and the
    sweep has no other way to tell the two apart.
    """
    adapter, cache, cfg = _sampler_setup(bits=16)
    prompt = torch.randint(0, VOCAB - 1, (1, BLOCK))
    dense = _dense_generate(adapter, prompt, cfg)

    cache.cfg.policy = "every_n"
    cache.cfg.refresh_every = 1
    out = cached_generate(adapter, prompt, cfg, cache, reuse_window=True,
                          rotary_fn=_rotary, attention_fn=_attention)

    assert torch.equal(out, dense)
    assert cache.stats.window_reuses == 0, "nothing should have been reused"
    assert cache.stats.window_refreshes > 0


def test_threshold_sampling_makes_the_step_count_an_observation():
    """Under a schedule the commits per step are an input; under a threshold
    they are what the model's confidence allows.

    That distinction is the whole reason this exists. Every damage number in
    the project is a share of decisions that moved, and the task grid shows
    those decisions do not change a single answer at any width. Steps are the
    one quantity left that a damaged cache can make worse in a way a user
    feels, and a schedule fixes it in advance so the cache cannot touch it.
    """
    from dllmquant.models.llada2_local import cached_generate

    adapter, cache, cfg = _sampler_setup(16)
    remove_block_cache(adapter.model)
    prompt = torch.randint(0, VOCAB - 1, (1, BLOCK))

    x = cached_generate(adapter, prompt, cfg, cache, reuse_window=True,
                        threshold=0.0, rotary_fn=_rotary,
                        attention_fn=_attention)

    # A threshold of zero commits every masked position at once, so one step
    # per block and nothing left masked.
    assert cache.stats.steps_used
    assert all(n == 1 for n in cache.stats.steps_used)
    assert (x[:, BLOCK:] != adapter.mask_id).all()


def test_an_impossible_threshold_still_commits_one_token_per_step():
    """A step that commits nothing cannot end, and the sampler would spin
    until the cap with the block still masked."""
    from dllmquant.models.llada2_local import cached_generate

    adapter, cache, cfg = _sampler_setup(16)
    remove_block_cache(adapter.model)
    prompt = torch.randint(0, VOCAB - 1, (1, BLOCK))

    x = cached_generate(adapter, prompt, cfg, cache, reuse_window=True,
                        threshold=2.0, rotary_fn=_rotary,
                        attention_fn=_attention)

    assert all(n == BLOCK for n in cache.stats.steps_used)
    assert (x[:, BLOCK:] != adapter.mask_id).all()
