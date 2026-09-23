"""A quantized prefix cache for Fast-dLLM-v2, without touching its vendored code.

Fast-dLLM-v2 drives its own cache: ``generate`` runs the block loop, and the prefix keys
and values are written once per block through a ``transformers`` ``DynamicCache`` that the
model instantiates itself inside ``forward``. There is no place to hand a cache in, so the
class the model constructs is what gets replaced here -- the same tactic as patching
``apply_rotary_pos_emb`` for R4, and for the same reason: the quantizer has to sit between
the projections and the attention call, and everything above that stays the checkpoint's.

What is quantized is exactly the store: the entries a finished block writes, at the width
and axis given, read back by every later block. The block being decoded is untouched, which
is what this checkpoint ships (``use_block_cache=False``) and what the staleness measurements
on LLaDA2.0-mini say is the only safe arrangement.

The axis names are this project's, and they are the grouping direction, not what the scale
is shared across: ``token`` is one scale per channel (KIVI's per-channel, the safe one for
keys) and ``channel`` is one scale per token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch

from ..cache import add_key_noise, quantize_kv


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rotated_key_bias(bias: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                     kv_heads: int) -> torch.Tensor:
    """The k_proj bias as it appears in the stored key: constant per channel, then rotated.

    RoPE is linear, so a key is ``RoPE(Wx) + RoPE(b)``. The second term is the same vector
    at every position before the rotation, and it is where Qwen2.5-family key outliers come
    from. Subtracting it before the quantizer and adding it back after is exact at any width
    and removes the parameter-induced part of the outlier -- the counterpart of the gain
    migration on a model that has no QK-Norm to migrate.
    """
    b = bias.view(1, kv_heads, 1, -1).to(cos.dtype)
    c = cos.unsqueeze(1)
    t = sin.unsqueeze(1)
    return b * c + _rotate_half(b) * t


@dataclass
class FDv2CacheStats:
    writes: int = 0
    entries: int = 0
    key_axis: str = ""
    value_axis: str = ""
    pre_bias: bool = False
    skipped: int = 0

    def describe(self) -> str:
        bias = ", K stored before the k_proj bias" if self.pre_bias else ""
        kept = f", {self.skipped} writes kept exact" if self.skipped else ""
        return (f"prefix cache: {self.writes} writes, {self.entries} entries stored, "
                f"K along {self.key_axis}, V along {self.value_axis}{bias}{kept}")


def make_quantized_cache_class(
    base,
    *,
    key_bits: int,
    value_bits: int,
    group_size: int,
    key_axis: str,
    value_axis: str,
    stats: FDv2CacheStats,
    key_biases: Optional[List[torch.Tensor]] = None,
    kv_heads: int = 0,
    clip_ratio: float = 0.95,
    value_group: int = 0,
    rotation=None,
    key_mean: bool = False,
    key_noise: float = 0.0,
    key_noise_mode: str = "iid",
    skip_layers: frozenset = frozenset(),
):
    """A ``DynamicCache`` subclass that rounds what it stores.

    ``base`` is passed in rather than imported so the class follows whatever
    ``DynamicCache`` the installed transformers provides -- the vendored model builds its
    cache from that name, and a subclass of a different one would not be accepted by it.
    """

    class QuantizedPrefixCache(base):
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            if layer_idx in skip_layers:
                # Kept in full precision. The per-layer profile of the attention
                # movement says the rotation's damage is two layers out of 28 --
                # the first and the last, at KL 3.2 and 5.6 where the other 26
                # sit at 0.02 -- so whether those two carry the whole collapse is
                # a question about them, not about the width.
                stats.skipped += 1
                return super().update(key_states, value_states, layer_idx, cache_kwargs)
            k = key_states.float()
            b_rot = None
            if key_biases is not None:
                # Per layer: every block has its own k_proj bias, and subtracting one
                # block's bias from another's keys would be a different tensor, not an
                # identity.
                key_bias = key_biases[layer_idx]
                cos, sin = (cache_kwargs or {}).get("cos"), (cache_kwargs or {}).get("sin")
                if cos is None or sin is None:
                    raise RuntimeError(
                        "pre-bias quantization needs the rotary tables the attention passes "
                        "in cache_kwargs; this revision does not provide them")
                b_rot = rotated_key_bias(key_bias, cos, sin, kv_heads).float()
                if b_rot.shape[-2] != k.shape[-2]:
                    b_rot = b_rot[..., -k.shape[-2]:, :]
                # With R4 installed the keys reaching the store are RoPE(Wx + b) H, so
                # the bias term to take out is RoPE(b) H, not RoPE(b). Subtracting the
                # unrotated one and adding it back is still an identity, but it leaves
                # the bias inside the quantizer's range -- a pre-bias label on a cell that
                # did nothing. The rotation is read at write time, from the patched rotary
                # function, so the two cannot disagree.
                h = rotation() if rotation is not None else None
                if h is not None:
                    b_rot = b_rot @ h.to(b_rot.device, b_rot.dtype)
                k = k - b_rot
            # SageAttention's "smooth K", applied to storage: the per-channel mean of the
            # keys this write stores (a block, or the prompt), taken out before the
            # quantizer and put back after, one vector per write kept alongside the codes.
            # Data-driven where pre-bias is exact: after RoPE the bias rotates with the
            # position, and a mean over positions keeps only the part that does not.
            m = k.mean(dim=-2, keepdim=True) if key_mean else None
            if m is not None:
                k = k - m
            k = quantize_kv(k, key_bits, group_size, axis=key_axis, clip_ratio=clip_ratio)
            if m is not None:
                k = k + m
            if b_rot is not None:
                k = k + b_rot
            # The structureless control: noise of a chosen size instead of, or on top
            # of, the quantizer's error, measured against the true keys so the dose
            # means the same at 16 bits as at four.
            k = add_key_noise(k, key_states.float(), key_noise, key_noise_mode)
            stats.pre_bias = key_biases is not None
            v = quantize_kv(value_states.float(), value_bits, value_group or group_size,
                            axis=value_axis, clip_ratio=clip_ratio)
            stats.writes += 1
            stats.entries += int(key_states.shape[-2])
            return super().update(
                k.to(key_states.dtype), v.to(value_states.dtype), layer_idx, cache_kwargs
            )

    QuantizedPrefixCache.__name__ = "QuantizedPrefixCache"
    return QuantizedPrefixCache


def install_quantized_cache(
    adapter,
    *,
    bits: int = 16,
    key_bits: int = 0,
    value_bits: int = 0,
    group_size: int = 128,
    key_axis: str = "token",
    value_axis: str = "channel",
    pre_bias: bool = False,
    clip_ratio: float = 0.95,
    value_group_size: int = 0,
    key_mean: bool = False,
    key_noise: float = 0.0,
    key_noise_mode: str = "iid",
    skip_layers: "tuple | None" = None,
) -> Tuple[Callable[[], None], FDv2CacheStats]:
    """Replace the cache class the vendored model constructs. Returns (remove, stats).

    ``pre_bias`` stores ``K`` with the rotated k_proj bias removed and adds it back on the
    way out: exact, and it takes the parameter-induced part of the key outlier out of the
    quantizer's range. On a model with no such bias it is refused rather than silently
    doing nothing.
    """
    if pre_bias and key_mean:
        raise ValueError("--pre-bias and --key-mean are two answers to one question; "
                         "compare them in separate runs")
    import sys

    mod = sys.modules[type(adapter.model).__module__]
    base = getattr(mod, "DynamicCache", None)
    if base is None:
        raise RuntimeError(
            f"{mod.__name__} does not build its cache from a module-level DynamicCache, "
            "so there is nothing to replace; check the vendored revision"
        )
    key_biases = None
    if pre_bias:
        from .base import find_submodule
        from .llada import _K_NAMES

        key_biases = []
        for i, block in enumerate(adapter.blocks):
            k_proj = find_submodule(block, _K_NAMES)
            bias = getattr(k_proj, "bias", None) if k_proj is not None else None
            if bias is None:
                raise RuntimeError(
                    f"--pre-bias on a model whose k_proj has no bias (block {i}): there is "
                    "nothing to take out of the quantizer's range, and a run labelled "
                    "pre-bias that did nothing would be worse than no run")
            key_biases.append(bias.detach())

    stats = FDv2CacheStats(key_axis=key_axis, value_axis=value_axis)

    def current_rotation():
        state = getattr(getattr(mod, "apply_rotary_pos_emb", None), "rotation_state", None)
        return None if not state else state.get("h")

    mod.DynamicCache = make_quantized_cache_class(
        base,
        key_bits=key_bits or bits,
        value_bits=value_bits or bits,
        group_size=group_size,
        key_axis=key_axis,
        value_axis=value_axis,
        stats=stats,
        key_biases=key_biases,
        kv_heads=adapter.n_kv_heads,
        clip_ratio=clip_ratio,
        value_group=value_group_size,
        rotation=current_rotation,
        key_mean=key_mean,
        key_noise=key_noise,
        key_noise_mode=key_noise_mode,
        skip_layers=frozenset(skip_layers or ()),
    )

    def remove() -> None:
        mod.DynamicCache = base

    return remove, stats


def fdv2_generate(
    adapter,
    prompt_ids: torch.Tensor,
    cfg,
    *,
    threshold: float = 0.95,
    block_size: int = 32,
    small_block_size: int = 8,
    top_p: float = 0.95,
    temperature: float = 0.0,
    use_block_cache: bool = False,
) -> torch.Tensor:
    """One question through the checkpoint's own sampler.

    The cache is whatever ``install_quantized_cache`` left in place, so this function has
    no quantization knobs of its own: the two must not be able to disagree.

    ``use_block_cache`` is the checkpoint's own switch for reusing the current block: its
    K/V are computed in full when a sub-block starts and, while that sub-block is decoded,
    only the sub-block's own positions are recomputed -- the rest of the block is read back
    stale. The block store is built from the same replaced ``DynamicCache`` class, so the
    entries written at the start of a sub-block are rounded like the prefix; the sub-block's
    own positions, written into that store by slice assignment inside attention, are the
    freshly recomputed ones and stay in full precision. What is stale is therefore also
    what is rounded, which is the combination the LLaDA2.0-mini staleness cells measured.
    """
    ids = prompt_ids if prompt_ids.dim() > 1 else prompt_ids.unsqueeze(0)
    ids = ids.to(adapter.model.device)
    stop = getattr(adapter.model.config, "eos_token_id", None)
    with torch.no_grad():
        return adapter.model.generate(
            ids,
            max_new_tokens=cfg.gen_length,
            tokenizer=adapter.tokenizer,
            mask_id=adapter.mask_id,
            threshold=threshold,
            small_block_size=small_block_size,
            block_size=block_size,
            stop_token=stop if isinstance(stop, int) else 151645,
            top_p=top_p,
            temperature=temperature,
            use_block_cache=use_block_cache,
        )


__all__ = ["install_quantized_cache", "make_quantized_cache_class", "fdv2_generate",
           "rotated_key_bias", "FDv2CacheStats"]
