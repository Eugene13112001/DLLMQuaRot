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
model.

**Two stores on two clocks, and the distinction is the experiment.** The prefix
rides the checkpoint's own ``layer_past``; ``stale_prefix`` lets the policy age
it, which is the axis unique to bidirectional attention. The current block
cannot go through ``layer_past`` at all -- its keys are built inside the
forward, after rotary, and ``layer_past`` only prepends -- so ``reuse_window``
hooks ``_scaled_dot_product_attention``, which sees q, k and v assembled, and
swaps the block's slice of k and v there. That second axis is the one LLaDA2.0
has, and without it the two families are measured on different objects: the
first grid run this way put damage at 3.0% against 63.7%, which compares a
prefix of settled tokens with a block that changes every step.
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


class WindowState:
    """Per-block switch for the current block's own K/V.

    The prefix goes through ``layer_past``, which the checkpoint already
    supports. The current block cannot: its keys are computed inside the
    forward, after rotary, and ``layer_past`` only prepends. So this hooks the
    one method that sees them assembled --
    ``LLaDABlock._scaled_dot_product_attention``, which receives q, k and v
    with rotary applied and the past already concatenated -- and swaps the
    block's slice of k and v before handing them on.

    Nothing is reimplemented: the original method is called with edited
    tensors, so the kernel choice, the bias handling and the dropout stay the
    checkpoint's own.
    """

    __slots__ = ("mode", "cache", "layer", "start", "width", "mask", "orig")

    def __init__(self) -> None:
        self.mode = "off"  # off | record | reuse
        self.cache: Optional[BlockKVCache] = None
        self.layer = 0
        # Where the block sits inside the assembled key axis. The forward
        # covers [lo, total) and the past covers [0, lo), so the block is the
        # first `width` keys after the past -- not the tail, which is the
        # masked suffix and must stay fresh.
        self.start = 0
        self.width = 0
        self.mask: Optional[torch.Tensor] = None
        self.orig: Optional[Callable] = None


# R4 for the dense path. The MoE side patches the checkpoint's module-level
# apply_rotary_pos_emb, but LLaDA-1.5's remote code has no such function, so
# there is no single call to wrap. This hook is the better place anyway: it
# sits after the block applied RoPE and *before* the window store writes K, so
# the store holds rotated keys, which is the whole point. A module-level holder
# rather than a field on WindowState because the states are built inside
# cached_generate, long after the caller asks for the rotation.
_QK_ROT: Optional[torch.Tensor] = None


def enable_qk_rotation(head_dim: int, seed: int = 0, device=None) -> None:
    """Turn R4 on for every block's attention on the dense path."""
    from ..algos.quarot import random_hadamard_matrix

    global _QK_ROT
    _QK_ROT = random_hadamard_matrix(head_dim, device=device, seed=seed)


def disable_qk_rotation() -> None:
    global _QK_ROT
    _QK_ROT = None


def _make_sdpa(state: "WindowState"):
    def sdpa(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
        global _QK_ROT
        if _QK_ROT is not None:
            if _QK_ROT.device != q.device:
                _QK_ROT = _QK_ROT.to(q.device)
            h = _QK_ROT.to(q.dtype)
            q, k = q @ h, k @ h.to(k.dtype)
        if state.mode != "off" and state.width > 0:
            a, b = state.start, state.start + state.width
            if state.mode == "record":
                kk, vv = state.cache.write_window(
                    state.layer,
                    k[..., a:b, :].contiguous(),
                    v[..., a:b, :].contiguous(),
                    mask=state.mask,
                )
            else:
                # Deliberately stale: computed before the tokens committed
                # since the last refresh. The queries stay fresh, so what is
                # measured is exactly the mismatch a reused block causes.
                kk, vv = state.cache.read_window(state.layer)
            k = torch.cat([k[..., :a, :], kk, k[..., b:, :]], dim=-2)
            v = torch.cat([v[..., :a, :], vv, v[..., b:, :]], dim=-2)
        return state.orig(q, k, v, attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal)
    return sdpa


def install_window_store(model: nn.Module, cache: BlockKVCache) -> List[WindowState]:
    """Hook every block's attention kernel, switched off."""
    import types

    states = []
    for i, block in enumerate(blocks(model)):
        state = WindowState()
        state.cache = cache
        state.layer = i
        state.orig = block._scaled_dot_product_attention
        block._dllm_window = state
        block._scaled_dot_product_attention = types.MethodType(
            _make_sdpa(state), block
        )
        states.append(state)
    return states


def remove_window_store(model: nn.Module) -> None:
    for block in blocks(model):
        block.__dict__.pop("_scaled_dot_product_attention", None)
        block.__dict__.pop("_dllm_window", None)


def set_window(
    states: List[WindowState],
    mode: str,
    start: int = 0,
    width: int = 0,
    mask: Optional[torch.Tensor] = None,
) -> None:
    for state in states:
        state.mode = mode
        state.start = start
        state.width = width
        state.mask = mask


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
    *,
    states: Optional[List["WindowState"]] = None,
    window: str = "off",
    width: int = 0,
    window_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Logits for positions ``lo..``, prefix from the store, block optional.

    ``window`` switches the current block's own K/V: ``off`` recomputes it
    every step, ``record`` recomputes and stores it, ``reuse`` reads the stored
    one back. Its offset inside the key axis is the prefix length, because the
    forward covers ``[lo, total)`` and the past covers ``[0, lo)``.
    """
    n_layers = len(blocks(model))
    if lo <= 0 or not cache.has(0):
        if states is not None:
            set_window(states, window, 0, width, window_mask)
        try:
            logits, _ = run_blocks(model, x, None)
        finally:
            if states is not None:
                set_window(states, "off")
        return logits[:, lo:, :]
    past = [cache.read(i) for i in range(n_layers)]
    if states is not None:
        set_window(states, window, lo, width, window_mask)
    try:
        logits, _ = run_blocks(model, x[:, lo:], past)
    finally:
        if states is not None:
            set_window(states, "off")
    return logits


@torch.no_grad()
def cached_generate(
    adapter,
    prompt_ids: torch.Tensor,
    cfg,
    cache: BlockKVCache,
    *,
    reuse_window: bool = True,
    stale_prefix: bool = False,
    threshold: Optional[float] = None,
    max_steps_per_block: int = 0,
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
    states = install_window_store(adapter.model, cache) if reuse_window else None
    try:
        for block_idx, (lo, hi) in enumerate(bounds):
            cache.reset_window()
            block_mask = x[:, lo:hi] == adapter.mask_id
            budget = get_num_transfer_tokens(block_mask, schedule[block_idx])

            # Two samplers, and the difference is the point -- the same
            # split as on the MoE path. Under the schedule the committed
            # count per step is an input, so parallelism cannot depend on
            # the cache. Under a threshold it is an observation, and a cache
            # that flattens confidence buys fewer commits per step.
            cap = max_steps_per_block or (hi - lo)
            n_steps = cap if threshold is not None else schedule[block_idx]
            used = 0
            for step in range(n_steps):
                still_masked = x[:, lo:hi] == adapter.mask_id
                if threshold is not None and not still_masked.any():
                    break
                used = step + 1
                ratio = float(still_masked.to(torch.float32).mean())

                # Two stores on two clocks, as on the MoE path. The prefix is
                # rewritten at the block boundary; letting the policy age it too
                # is a second axis, reachable with stale_prefix, and it is not the
                # one that compares to LLaDA2.0.
                stale = stale_prefix and not cache.should_refresh(
                    0, cache.step, ratio, block_boundary=(step == 0)
                )
                if stale:
                    # Counted here rather than in the store: the store cannot see
                    # a read it was never asked for, and a reused prefix is
                    # exactly a read that did not happen.
                    cache.stats.reuses += 1
                    cache.stats.ages.append(cache.age(0))
                elif step == 0 or stale_prefix:
                    refresh_prefix(adapter.model, cache, x, lo, adapter.mask_id)

                # One decision for every layer: the policies are uniform, so layer
                # 0 speaks for all of them.
                window = "off"
                if reuse_window:
                    window = ("record"
                          if cache.should_refresh_window(0, cache.step, ratio)
                          else "reuse")

                logits = logits_from_lo(
                    adapter.model, cache, x, lo,
                    states=states, window=window, width=hi - lo,
                    window_mask=still_masked if window == "record" else None,
                )[:, : hi - lo, :]
                cache.advance()

                probs = torch.softmax(logits.to(torch.float32), dim=-1)
                confidence, proposal = probs.max(dim=-1)

                if threshold is None and (int(budget[0, step]) <= 0
                                          or not still_masked.any()):
                    continue

                score = torch.where(
                    still_masked, confidence, torch.full_like(confidence, -torch.inf)
                )
                if threshold is None:
                    k = int(budget[0, step])
                    take = min(k, int(still_masked.sum()))
                    chosen = torch.topk(score[0], k=take).indices
                else:
                    # Everything the model is sure enough about, and never
                    # nothing: a step committing no token cannot end, and the
                    # sampler would spin to the cap.
                    chosen = (score[0] >= threshold).nonzero(as_tuple=True)[0]
                    if chosen.numel() == 0:
                        chosen = score[0].argmax().reshape(1)
                x[0, lo + chosen] = proposal[0, chosen]

                if on_step is not None:
                    on_step(block_idx, step, lo, hi, logits, x)

            if threshold is not None:
                cache.stats.steps_used.append(used)
    finally:
        if states is not None:
            remove_window_store(adapter.model)

    return x


__all__ = [
    "WindowState",
    "install_window_store",
    "remove_window_store",
    "set_window",
    "blocks",
    "bidirectional_bias",
    "cached_generate",
    "logits_from_lo",
    "refresh_prefix",
    "run_blocks",
]
