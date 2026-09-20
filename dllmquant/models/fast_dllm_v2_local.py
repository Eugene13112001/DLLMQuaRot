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

from ..cache import quantize_kv


@dataclass
class FDv2CacheStats:
    writes: int = 0
    entries: int = 0
    key_axis: str = ""
    value_axis: str = ""

    def describe(self) -> str:
        return (f"prefix cache: {self.writes} writes, {self.entries} entries stored, "
                f"K along {self.key_axis}, V along {self.value_axis}")


def make_quantized_cache_class(
    base,
    *,
    key_bits: int,
    value_bits: int,
    group_size: int,
    key_axis: str,
    value_axis: str,
    stats: FDv2CacheStats,
):
    """A ``DynamicCache`` subclass that rounds what it stores.

    ``base`` is passed in rather than imported so the class follows whatever
    ``DynamicCache`` the installed transformers provides -- the vendored model builds its
    cache from that name, and a subclass of a different one would not be accepted by it.
    """

    class QuantizedPrefixCache(base):
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            k = quantize_kv(key_states.float(), key_bits, group_size, axis=key_axis)
            v = quantize_kv(value_states.float(), value_bits, group_size, axis=value_axis)
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
) -> Tuple[Callable[[], None], FDv2CacheStats]:
    """Replace the cache class the vendored model constructs. Returns (remove, stats)."""
    import sys

    mod = sys.modules[type(adapter.model).__module__]
    base = getattr(mod, "DynamicCache", None)
    if base is None:
        raise RuntimeError(
            f"{mod.__name__} does not build its cache from a module-level DynamicCache, "
            "so there is nothing to replace; check the vendored revision"
        )
    stats = FDv2CacheStats(key_axis=key_axis, value_axis=value_axis)
    mod.DynamicCache = make_quantized_cache_class(
        base,
        key_bits=key_bits or bits,
        value_bits=value_bits or bits,
        group_size=group_size,
        key_axis=key_axis,
        value_axis=value_axis,
        stats=stats,
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
) -> torch.Tensor:
    """One question through the checkpoint's own sampler.

    The cache is whatever ``install_quantized_cache`` left in place, so this function has
    no quantization knobs of its own: the two must not be able to disagree.
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
        )


__all__ = ["install_quantized_cache", "make_quantized_cache_class", "fdv2_generate",
           "FDv2CacheStats"]
