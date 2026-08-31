"""Prefix-cached sampler for the dense LLaDA family (LLaDA-1.5).

The point of this file is that LLaDA-1.5 ages its cache for a *different
reason* than LLaDA2.0 does, which is what makes it the out-of-sample test the
composition law needs.

Under LLaDA2.0's block-causal mask a closed position cannot see the block being
decoded, so its K/V is exact by construction and the only staleness is inside
the current block. LLaDA-1.5 attends bidirectionally over the whole sequence:
every prefix position attends to the masked tail, the tail changes at every
commit, and so the prefix goes stale on its own and the error accumulates with
depth (measured: 0.029 at layer 1 rising to 0.279 at delay 4). Same two errors,
opposite regime. If the quadrature and the kappa identity survive both, they
describe a mechanism rather than one checkpoint.

**Why this does not go through the checkpoint's own forward.** The remote code
implements the whole cache path -- ``layer_past`` per block, ``past_length``,
the bidirectional bias sized at ``past_length + seq_len`` -- and then refuses
to run it:

    assert (past_key_values is None and not use_cache), \\
        "The kvcache is not suppotred for MDM."

The refusal is right in general and wrong for the one case used here. With the
cache holding exactly positions ``0..lo-1`` the autoregressive bookkeeping is
correct by accident: ``past_length == lo``, so the window's rotary positions
come out at ``lo..`` and the bias is sized right. Rather than vendor and patch
fifteen hundred lines of someone else's file to remove one assert, the block
loop is driven directly here -- the same choice already made in
``llada2_local`` and for the same reason.

**What is forwarded.** The window is ``[lo, total)`` -- the current block *and*
the masked tail -- not just the block. Bidirectional attention means the block's
queries read the tail, so dropping it would not be staleness but a different
model. The saving and the staleness both sit in the prefix, which is where
LLaDA-1.5 accumulates its drift anyway.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cache import BlockKVCache
from .base import _split_evenly, get_num_transfer_tokens


def _transformer(model: nn.Module) -> nn.Module:
    """The container holding ``wte``, ``blocks`` and ``ln_f``.

    Probed rather than hard-coded: the checkpoint exposes the inner model at
    ``model.model`` behind the LM head on some revisions and at the top level
    on others, and a wrong guess fails deep inside the block loop with an
    error about attributes rather than about layout.
    """
    for candidate in (model, getattr(model, "model", None)):
        if candidate is not None and hasattr(candidate, "transformer"):
            return candidate.transformer
    raise AttributeError(
        "no .transformer under this model -- expected the dense LLaDA layout "
        "(transformer.wte / .blocks / .ln_f)"
    )


def blocks(model: nn.Module) -> List[nn.Module]:
    """The transformer blocks, flattened out of block groups if grouped."""
    tr = _transformer(model)
    if hasattr(tr, "blocks"):
        return list(tr.blocks)
    grouped = []
    for group in tr.block_groups:
        grouped.extend(list(group))
    return grouped


def _embed(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Token embeddings, with the checkpoint's optional scaling and dropout.

    No positional embedding branch: that path exists in the remote code only
    for ``not (alibi or rope)``, and this family uses rope, which is applied
    inside each block's attention.
    """
    tr = _transformer(model)
    x = tr.wte(input_ids)
    cfg = model.config
    if getattr(cfg, "input_emb_norm", False):
        x = x * (cfg.d_model ** 0.5)
    return tr.emb_drop(x)


def _head(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Final norm and unembedding, tied or untied."""
    tr = _transformer(model)
    x = tr.ln_f(x)
    ff_out = getattr(tr, "ff_out", None)
    if ff_out is not None:
        return ff_out(x)
    return F.linear(x, tr.wte.weight)


def bidirectional_bias(k_len: int, dtype: torch.dtype, device) -> torch.Tensor:
    """The additive bias for unmasked attention, as a full ``k_len`` square.

    Square, not pre-sliced to the window, because the block slices it itself:
    its attention takes ``[key_len - query_len : key_len, :key_len]``, the
    usual convention for a bias handed down whole from the model's forward.
    Passing the already-narrow ``(1, 1, q_len, k_len)`` makes that slice read
    past the end and the error surfaces inside SDPA as a shape mismatch that
    names neither the bias nor the window.

    All zeros, matching ``get_bidirectional_attention_bias``. Passing ``None``
    instead would leave the choice of mask to the block, and its default is
    not guaranteed to stay non-causal across revisions.
    """
    return torch.zeros((1, 1, k_len, k_len), dtype=dtype, device=device)


@torch.no_grad()
def run_blocks(
    model: nn.Module,
    input_ids: torch.Tensor,
    past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    *,
    collect: bool = False,
) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
    """Embedding, blocks, head -- the model's own forward minus the assert.

    ``past_kv`` holds one (K, V) pair per block for positions ``0..lo-1``,
    already carrying their rotary phase from when they were computed. Each
    block concatenates them ahead of the fresh keys, and its rotary sees a key
    length of ``lo + window`` against a query length of ``window``, which is
    exactly the offset the window's positions need.

    With ``collect`` the fresh K/V of every block are returned, which is how
    the store is filled.
    """
    x = _embed(model, input_ids)
    q_len = input_ids.shape[1]
    past_len = 0 if past_kv is None else int(past_kv[0][0].shape[-2])
    bias = bidirectional_bias(past_len + q_len, x.dtype, x.device)

    harvested: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i, block in enumerate(blocks(model)):
        layer_past = None if past_kv is None else past_kv[i]
        x, present = block(
            x, attention_bias=bias, layer_past=layer_past, use_cache=collect
        )
        if collect:
            harvested.append(present)
    return _head(model, x), (harvested if collect else None)


@torch.no_grad()
def refresh_prefix(
    model: nn.Module,
    cache: BlockKVCache,
    x: torch.Tensor,
    lo: int,
    mask_id: int,
) -> None:
    """Recompute the whole sequence and store the prefix K/V, quantized.

    The full pass is the expensive thing this cache exists to avoid, so it runs
    only when the policy says the prefix has aged out. What is stored is the
    slice ``0..lo``: the window is recomputed every step regardless, and
    keeping its keys would only let them be read at the wrong positions later.
    """
    _, harvested = run_blocks(model, x, None, collect=True)
    if lo <= 0:
        return
    mask = (x[:, :lo] == mask_id)
    before = cache.stats.refreshes
    for layer, (k, v) in enumerate(harvested):
        cache.write(
            layer,
            k[..., :lo, :].contiguous(),
            v[..., :lo, :].contiguous(),
            mask=mask,
        )
    # One refresh, not thirty-two. `write` counts per layer, which is the
    # right unit on the MoE path where each layer decides for itself, but
    # here every layer is rewritten together and the reuse counter opposite
    # it ticks once per step. Left uncorrected the hit rate divides layers by
    # steps and reports a number that is not a rate of anything -- and the
    # budget axis in `frontier.py` is computed from exactly that.
    cache.stats.refreshes = before + 1


@torch.no_grad()
def logits_from_lo(
    model: nn.Module,
    cache: BlockKVCache,
    x: torch.Tensor,
    lo: int,
) -> torch.Tensor:
    """Logits for positions ``lo..`` reading the prefix out of the store."""
    if lo <= 0 or not cache.has(0):
        logits, _ = run_blocks(model, x, None)
        return logits[:, lo:, :]
    past = [cache.read(i) for i in range(len(blocks(model)))]
    logits, _ = run_blocks(model, x[:, lo:], past)
    return logits


@torch.no_grad()
def cached_generate(
    adapter,
    prompt_ids: torch.Tensor,
    cfg,
    cache: BlockKVCache,
    *,
    reuse_window: bool = True,
    on_step: Optional[Callable] = None,
):
    """The semi-autoregressive sampler with a quantized, ageing prefix cache.

    Deliberately mirrors ``llada2_local.cached_generate`` in shape and in the
    meaning of its arguments so the same measurement scripts drive both, but
    ``reuse_window`` means something different here and the difference is the
    point. On LLaDA2.0 the prefix is exact by construction, so staleness can
    only be injected into the current block and the flag says whether to cache
    that block. Here it is the *prefix* that ages, so the flag says whether to
    let it: ``False`` refreshes every step, which is the exact variant, and
    ``True`` follows the configured policy.

    A 16-bit cache refreshed every step must reproduce the uncached sampler
    token for token; that equivalence is what makes anything measured off this
    path arguable in the first place.
    """
    device = next(adapter.model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    p_len = prompt_ids.shape[1]
    total, bounds = adapter._sequence_layout(p_len, cfg)

    x = torch.full((1, total), adapter.mask_id, dtype=torch.long, device=device)
    x[:, :p_len] = prompt_ids

    if cfg.steps < len(bounds):
        raise ValueError(
            f"{cfg.steps} steps cannot cover {len(bounds)} blocks -- some block "
            "would never be visited and stay masked"
        )
    schedule = _split_evenly(cfg.steps, len(bounds))

    cache.reset()
    for block_idx, (lo, hi) in enumerate(bounds):
        block_mask = x[:, lo:hi] == adapter.mask_id
        budget = get_num_transfer_tokens(block_mask, schedule[block_idx])

        for step in range(schedule[block_idx]):
            still_masked = x[:, lo:hi] == adapter.mask_id
            ratio = float(still_masked.to(torch.float32).mean())

            # One decision for every layer: the policies are uniform, so layer
            # 0 speaks for all of them, as on the MoE path.
            stale = reuse_window and not cache.should_refresh(
                0, cache.step, ratio, block_boundary=(step == 0)
            )
            if stale:
                # Counted here rather than in the store: the store cannot see
                # a read it was never asked for, and a reused prefix is
                # exactly a read that did not happen.
                cache.stats.reuses += 1
                cache.stats.ages.append(cache.age(0))
            else:
                refresh_prefix(adapter.model, cache, x, lo, adapter.mask_id)

            logits = logits_from_lo(adapter.model, cache, x, lo)[:, : hi - lo, :]
            cache.advance()

            probs = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence, proposal = probs.max(dim=-1)

            k = int(budget[0, step])
            if k <= 0 or not still_masked.any():
                continue

            score = torch.where(
                still_masked, confidence, torch.full_like(confidence, -torch.inf)
            )
            take = min(k, int(still_masked.sum()))
            chosen = torch.topk(score[0], k=take).indices
            x[0, lo + chosen] = proposal[0, chosen]

            if on_step is not None:
                on_step(block_idx, step, lo, hi, logits, x)

    return x


__all__ = [
    "blocks",
    "bidirectional_bias",
    "cached_generate",
    "logits_from_lo",
    "refresh_prefix",
    "run_blocks",
]
