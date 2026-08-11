"""What this project changes about LLaDA2.0, kept out of the vendored copy.

The checkpoint's own code is in ``dllmquant/vendor/llada2_moe`` and is not
edited (see its PROVENANCE.md). Everything here is applied to a loaded model
at runtime.

Why the model's own cache cannot be used
----------------------------------------
``LLaDA2MoeAttention`` already threads ``past_key_value`` through to
``DynamicCache.update()``, so at first glance the cache is a matter of passing
one in.  It is not: ``LLaDA2MoeModel.forward`` demands an attention mask of
exactly ``(batch, 1, T, T)`` where ``T`` is the length of *this* forward's
input, and refuses anything else.  With a cache the query is shorter than the
keys, so the mask is not square and the model raises before attention runs.
The plumbing is there; the entry point is closed.

What block diffusion gives for free
-----------------------------------
Attention here is block-causal: a query attends to its own block and to every
earlier one, never to a later one.  So while block ``B`` is being decoded, a
position in an earlier block attends only to positions in blocks it could
already see, none of which change -- its tokens are committed and its inputs
are fixed.  **Its K/V are constant for the whole of block B's decoding.**

That splits the cache into two halves with completely different error
characters, and the split is the point of the experiment:

* the *prefix* (blocks before the current one) is exactly reusable. Cache it
  and the only error is the one quantization puts there. Correctness is
  checkable against a full forward, bit for bit, with no reference to quality.
* the *current block* is not: its tokens change at every step, so reusing its
  K/V trades staleness for compute, and that trade is what the refresh
  policies in :mod:`dllmquant.cache` exist to explore.

This module implements the first half. The second changes what the model
computes and belongs behind its own measurement.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

from ..cache import BlockKVCache


class CacheState:
    """Per-attention-layer switch, flipped by the runner around a forward.

    ``off`` reproduces the vendored attention exactly, so installing the cache
    and never enabling it is a no-op -- which is what makes the equivalence
    test meaningful.
    """

    __slots__ = ("mode", "cache", "prefix_len")

    def __init__(self) -> None:
        self.mode = "off"  # off | record | use
        self.cache: Optional[BlockKVCache] = None
        self.prefix_len = 0


def _default_kernels():
    """The model's own rotary and attention functions.

    Taken from the vendored module rather than reimplemented: a second
    implementation of partial RoPE would be a second thing to keep in step,
    and its divergence would show up as a quality difference nobody could
    trace. Imported lazily -- this file must stay importable under the older
    transformers that LLaDA-1.5 runs on.
    """
    from ..vendor.llada2_moe.modeling_llada2_moe import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    return apply_rotary_pos_emb, eager_attention_forward


def _make_forward(rotary_fn: Callable, attention_fn: Callable):
    """Build the replacement ``forward`` for one attention module."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        state: CacheState = self._dllm_cache_state
        bsz, q_len, _ = hidden_states.size()

        qkv = self.query_key_value(hidden_states).view(
            bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim
        )
        query, key, value = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads],
            dim=-2,
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if getattr(self.config, "use_qk_norm", False):
            query = self.query_layernorm(query)
            key = self.key_layernorm(key)

        cos, sin = position_embeddings
        query, key = rotary_fn(query, key, cos, sin)

        # The cache holds K/V *after* RoPE, which is the only point where they
        # are reusable: RoPE depends on absolute position, and a cached entry
        # keeps the position it was computed at.
        if state.mode == "record":
            state.cache.write(
                self.layer_idx,
                key[..., : state.prefix_len, :].contiguous(),
                value[..., : state.prefix_len, :].contiguous(),
            )
        elif state.mode == "use":
            past_key, past_value = state.cache.read(self.layer_idx)
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        attn_output, attn_weights = attention_fn(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        return self.dense(attn_output), attn_weights, past_key_value

    return forward


def attention_modules(model: nn.Module) -> List[nn.Module]:
    return [layer.attention for layer in model.model.layers]


def install_block_cache(
    model: nn.Module,
    cache: BlockKVCache,
    *,
    rotary_fn: Optional[Callable] = None,
    attention_fn: Optional[Callable] = None,
) -> List[CacheState]:
    """Give every attention layer a cache-aware forward, switched off.

    The forward is bound per instance rather than installed on the class or
    wrapped in a new module: the module tree keeps its exact shape and every
    qualified name stays what it was.  A wrapper here would rename
    ``layers.N.attention.query_key_value``, and the naming is load-bearing --
    the rotation plan, the skip lists and the IA-AQ probe all key off it.
    """
    import types

    if rotary_fn is None or attention_fn is None:
        default_rotary, default_attention = _default_kernels()
        rotary_fn = rotary_fn or default_rotary
        attention_fn = attention_fn or default_attention

    forward = _make_forward(rotary_fn, attention_fn)
    states = []
    for attn in attention_modules(model):
        state = CacheState()
        state.cache = cache
        attn._dllm_cache_state = state
        attn.forward = types.MethodType(forward, attn)
        states.append(state)
    return states


def remove_block_cache(model: nn.Module) -> None:
    """Restore the vendored forward on every attention layer."""
    for attn in attention_modules(model):
        attn.__dict__.pop("forward", None)
        attn.__dict__.pop("_dllm_cache_state", None)


def _set_mode(states: List[CacheState], mode: str, prefix_len: int = 0) -> None:
    for state in states:
        state.mode = mode
        state.prefix_len = prefix_len


@torch.no_grad()
def forward_window(
    model: nn.Module,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the stack over a slice of positions and return its logits.

    Reimplements ``LLaDA2MoeModel.forward`` only as far as necessary: the
    decoder layers are called as they are, so everything inside a block --
    norms, routing, experts, residuals -- is the checkpoint's own code. What
    is replaced is the part that builds the mask and refuses a non-square one.
    """
    inner = model.model
    hidden = inner.word_embeddings(input_ids)
    position_embeddings = inner.rotary_emb(hidden, position_ids)

    for layer in inner.layers:
        hidden = layer(
            hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )[0]

    return model.lm_head(inner.norm(hidden))


def block_causal_mask(
    seq_len: int,
    block_length: int,
    *,
    rows: Optional[Tuple[int, int]] = None,
    batch_size: int = 1,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Additive block-causal mask, optionally only for a range of queries.

    ``rows=(lo, hi)`` returns ``[B, 1, hi - lo, seq_len]``: the queries of one
    window against every key up to ``seq_len``. That rectangle is exactly what
    the vendored model refuses to accept and exactly what a cache needs.
    """
    n_blocks = (seq_len + block_length - 1) // block_length
    allowed = torch.tril(torch.ones(n_blocks, n_blocks, device=device))
    expanded = allowed.repeat_interleave(block_length, dim=0).repeat_interleave(
        block_length, dim=1
    )[:seq_len, :seq_len]
    if rows is not None:
        expanded = expanded[rows[0] : rows[1], :]
    mask = expanded.log().to(dtype)
    return mask.unsqueeze(0).unsqueeze(0).expand(
        batch_size, 1, mask.shape[0], mask.shape[1]
    )


@torch.no_grad()
def refresh_prefix(
    model: nn.Module,
    states: List[CacheState],
    x: torch.Tensor,
    prefix_len: int,
    block_length: int,
) -> None:
    """Recompute and store the K/V of everything before the current block.

    One full forward, once per block boundary, against many windowed forwards
    -- which is the whole saving. The prefix is written through the cache's
    quantizer, so what later steps read back is what a real deployment would
    have kept, not the exact tensor.
    """
    total = x.shape[-1]
    mask = block_causal_mask(
        total, block_length, batch_size=x.shape[0],
        device=x.device, dtype=next(model.parameters()).dtype,
    )
    position_ids = torch.arange(total, device=x.device).unsqueeze(0)

    _set_mode(states, "record", prefix_len)
    try:
        forward_window(model, x, position_ids, mask)
    finally:
        _set_mode(states, "off")


@torch.no_grad()
def logits_for_window(
    model: nn.Module,
    states: List[CacheState],
    x: torch.Tensor,
    lo: int,
    hi: int,
    block_length: int,
) -> torch.Tensor:
    """Logits for positions ``[lo, hi)``, reading the prefix from the cache."""
    dtype = next(model.parameters()).dtype
    mask = block_causal_mask(
        hi, block_length, rows=(lo, hi), batch_size=x.shape[0],
        device=x.device, dtype=dtype,
    )
    position_ids = torch.arange(lo, hi, device=x.device).unsqueeze(0)

    _set_mode(states, "use", lo)
    try:
        return forward_window(model, x[:, lo:hi], position_ids, mask)
    finally:
        _set_mode(states, "off")


__all__ = [
    "CacheState",
    "install_block_cache",
    "remove_block_cache",
    "forward_window",
    "block_causal_mask",
    "refresh_prefix",
    "logits_for_window",
    "attention_modules",
]
