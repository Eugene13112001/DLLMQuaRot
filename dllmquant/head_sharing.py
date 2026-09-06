"""Does GQA's fan-out make K's rounding error *correlated*, or just shared?

Section 2.7.8 leaves the five-to-thirteenfold gap in the rounding floor
between LLaDA2.0-mini and LLaDA-1.5 unattributed. Routing was ruled out --
quantizing K alone and V alone damage equally, and V never enters the
softmax -- and what is left is the architectural difference: four K heads
read by sixteen query heads on the MoE, one to one on the dense model.

The mechanism that would make that matter is not the fan-out itself but what
the fan-out does to the *structure* of the error. Under MHA each query head
quantizes its own K and the per-head errors partly cancel when the heads are
summed. Under GQA four query heads read one stored row and see the same
error, which cannot cancel.

That is a claim about correlation at fixed magnitude, so the probe has to
vary correlation and hold magnitude. Two rows, same width, same grouping:

    shared        round the four stored heads, then replicate
    independent   replicate to sixteen, then round each on its own

**Stochastic rounding is what makes the second row mean anything.** Round
deterministically and two identical copies land on identical values, so
replicating first would still hand every reader the same error and the two
rows would differ in nothing at all. Adding a uniform offset before the
floor keeps the error's size and makes it independent per copy.

The probe is not a deployable configuration -- the independent row would
cost four times the storage. It exists to say whether correlation across
heads is the thing, which no configuration can answer.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


@torch.no_grad()
def stochastic_round(
    x: torch.Tensor,
    bits: int,
    group_size: int = 0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Fake-quantize along channels, rounding with a uniform offset.

    Groups run along the last axis, ``group_size`` channels of one token of
    one head -- the ``"channel"`` axis of ``quantize_kv``. Which axis is used
    is held fixed across the probe's two rows and is not what it varies; the
    axis question is answered separately in 2.1.

    ``group_size`` of 0 means one group per head, the whole ``head_dim``.
    """
    if bits >= 16:
        return x
    shape = x.shape
    g = group_size if group_size > 0 else shape[-1]
    if shape[-1] % g:
        raise ValueError(f"head_dim {shape[-1]} is not a multiple of {g}")

    xg = x.reshape(*shape[:-1], shape[-1] // g, g).float()
    lo = xg.amin(-1, keepdim=True)
    hi = xg.amax(-1, keepdim=True)
    n = 2 ** bits - 1
    scale = ((hi - lo) / n).clamp_min(1e-8)
    u = torch.rand(xg.shape, device=xg.device, dtype=xg.dtype,
                   generator=generator)
    q = torch.floor((xg - lo) / scale + u).clamp_(0, n)
    return (q * scale + lo).reshape(shape).to(x.dtype)


def repeat_heads(x: torch.Tensor, rep: int) -> torch.Tensor:
    """[B, H, T, D] -> [B, H*rep, T, D], each head repeated ``rep`` times.

    The same expansion transformers' ``repeat_kv`` performs, written out here
    because the probe needs it *before* attention rather than inside it.
    """
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, rep, t, d).reshape(b, h * rep, t, d)


def make_head_sharing_attention(
    inner: Callable,
    mode: str,
    bits: int,
    group_size: int = 0,
    seed: int = 0,
) -> Callable:
    """Wrap an attention kernel so K is quantized either side of the fan-out.

    Both rows replicate explicitly and set ``num_key_value_groups`` to 1 for
    the inner call, so the two differ in the order of two operations and in
    nothing else -- not in the width, not in the grouping, not in how many
    heads attention finally sees.
    """
    if mode not in ("shared", "independent"):
        raise ValueError(f"mode must be shared or independent, got {mode!r}")
    gen = {"g": None, "seed": seed}

    def fn(module, query, key, value, attention_mask, **kwargs):
        rep = int(getattr(module, "num_key_value_groups", 1) or 1)
        if rep <= 1:
            # MHA has no fan-out to decorrelate; the probe is a no-op and
            # saying so is better than quantizing one side only.
            return inner(module, query, key, value, attention_mask, **kwargs)

        if gen["g"] is None or gen["g"].device != key.device:
            gen["g"] = torch.Generator(device=key.device)
            gen["g"].manual_seed(gen["seed"])

        if mode == "shared":
            key = repeat_heads(stochastic_round(key, bits, group_size, gen["g"]), rep)
        else:
            key = stochastic_round(repeat_heads(key, rep), bits, group_size, gen["g"])
        value = repeat_heads(value, rep)

        saved = module.num_key_value_groups
        module.num_key_value_groups = 1
        try:
            return inner(module, query, key, value, attention_mask, **kwargs)
        finally:
            module.num_key_value_groups = saved

    return fn


__all__ = ["stochastic_round", "repeat_heads", "make_head_sharing_attention"]
