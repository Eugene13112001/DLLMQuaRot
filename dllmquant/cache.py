"""Quantized KV cache for masked diffusion LMs.

A diffusion LM has no exact KV cache: attention is bidirectional and any token
may change at the next denoising step, so keys and values computed one step ago
are already wrong.  Block-wise decoders (Fast-dLLM, LLaDA2.0) cache anyway and
accept the error, refreshing periodically.

That makes cache quantization here a *two-error* problem, unlike the
autoregressive case:

    staleness    grows with time since the entry was refreshed
    rounding     fixed at write time, but never averages out (see below)

and the refresh interval is a single knob that resets both.  Whether the two
errors add or one hides the other is an open question -- ``CacheStats`` records
the age of every entry so the two can be separated in analysis.

Three things here are specific to diffusion and have no autoregressive analogue:

1. **Entries have a status.**  A decoded position's K/V is final; a masked
   position's will be overwritten as soon as the token is committed.  Spending
   precision on the second is spending it on something about to be discarded,
   so bit width can follow mask status.

2. **Rounding errors are correlated.**  At a high mask ratio nearly every
   position carries the same embedding row (the ``[MASK]`` token), so their
   K/V are near-identical and round the same way.  In ``out_i = Σ_j A_ij v_j``
   independent errors partially cancel; identical ones add coherently.  The
   usual "quantization noise averages over positions" argument fails.

3. **Refresh can be adaptive for free.**  The mask ratio is known at every
   step at no cost, and it says how fast the cache is going stale: everything
   moves early in the trajectory, almost nothing moves late.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class KVCacheConfig:
    """Numeric format and refresh policy of the cache."""

    enabled: bool = False

    # --- format ---------------------------------------------------------
    # Bits for entries at positions that are already decoded.  Those entries
    # are final, so precision spent on them is kept.
    decoded_bits: int = 4
    # Bits for entries at masked positions.  They are overwritten as soon as
    # the token is committed, so precision there has a short shelf life.
    # Equal to decoded_bits by default -- the asymmetry is an experiment, not
    # an assumption.
    masked_bits: int = 4
    # Override the bit width for one side of the cache. K and V are not
    # symmetric consumers of their own error: K goes into the softmax, where a
    # perturbation is exponentiated and reshapes the whole attention
    # distribution, while V is summed against weights that are already fixed.
    # Left as None both sides follow the status-based widths above, which is
    # what every implementation here did before this was measurable.
    key_bits: Optional[int] = None
    value_bits: Optional[int] = None
    # Group size along head_dim. 128 is QuaRot's KV setting, and the
    # per-token-across-all-channels alternative measured 52 GSM8K points worse
    # on this model's activations, so grouping is not optional here.
    group_size: int = 128
    # Which direction a group runs in, separately for the two sides. See
    # quantize_kv: "channel" is KIVI's per-token, "token" is KIVI's
    # per-channel. KIVI's finding is K wants "token" and V wants "channel";
    # both default to "channel" here only because every number measured before
    # this existed was taken that way and has to stay reproducible.
    key_axis: str = "channel"
    value_axis: str = "channel"
    symmetric: bool = False
    clip_ratio: float = 0.95
    # Calibrated per-channel scales, selected by the mask ratio of what is
    # being stored. Set means the scales stop travelling with the cache: the
    # group size and the axis no longer apply, because a static scale is
    # per-channel by construction, and the reported bit width becomes the true
    # one. Left None the cache quantizes dynamically, which is what every
    # number measured before this existed did.
    scale_book: Optional["StaticScaleBook"] = None
    # Which sides the book applies to. K and V need not be answered together:
    # the token axis won for K because its outliers sit in fixed channels, and
    # that is exactly the property a calibrated scale relies on -- V never had
    # it, and made almost no difference either way. Naming the sides keeps
    # "static K, dynamic V" expressible instead of assumed.
    scale_book_kinds: Tuple[str, ...] = ("key", "value")
    # Rank of a correction added on top of the quantized tensor, 0 for none.
    # K only by default: whatever the residual's structure turns out to be, K
    # is the side whose error is exponentiated by the softmax, and spending the
    # same budget on both halves the rank each gets.
    lowrank_rank: int = 0
    lowrank_kinds: Tuple[str, ...] = ("key",)
    lowrank_factor_bits: int = 16
    # Extra precision for the widest channels of each head -- the competitor a
    # low-rank correction has to beat when the residual's dominant direction is
    # a single channel, which is what it measured out to be. Which channels
    # those are is a property of the layer, so the index list lives in the
    # model and the cache carries nothing for it.
    wide_channels: int = 0
    wide_extra_bits: int = 0
    wide_kinds: Tuple[str, ...] = ("key",)

    # --- refresh policy --------------------------------------------------
    # 'never'      keep entries until the block boundary (pure staleness)
    # 'every_n'    recompute every `refresh_every` denoising steps
    # 'block'      recompute at block boundaries only
    # 'mask_ratio' recompute more often while the sequence is mostly masked
    policy: str = "every_n"
    refresh_every: int = 4
    # For 'mask_ratio': interval = max(1, round(min_interval + (max_interval -
    # min_interval) * (1 - mask_ratio))).  Fully masked -> refresh every step;
    # fully decoded -> refresh rarely.
    min_interval: int = 1
    max_interval: int = 16
    # A `refresh_masked_only` flag stood here, declared and never read by
    # anything. Selective refresh is a real direction and it is not this
    # field's to hold: Polestar (2607.14107) already ranks positions by
    # representation drift, and 2.7.7 says what the right criterion would be --
    # an entry's contribution to the deficit, which the law makes additive.
    # A field that does nothing is a trap for whoever reads the config next.

    def __post_init__(self) -> None:
        valid = {"never", "every_n", "block", "mask_ratio"}
        if self.policy not in valid:
            raise ValueError(f"policy must be one of {sorted(valid)}")
        for name in ("decoded_bits", "masked_bits", "key_bits", "value_bits"):
            b = getattr(self, name)
            if b is not None and not 2 <= b <= 16:
                raise ValueError(f"{name} must be in [2, 16], got {b}")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        for name in ("key_axis", "value_axis"):
            if getattr(self, name) not in GROUP_AXES:
                raise ValueError(
                    f"{name} must be one of {GROUP_AXES}, "
                    f"got {getattr(self, name)!r}"
                )
        if not 0 < self.clip_ratio <= 1.0:
            raise ValueError("clip_ratio must be in (0, 1]")
        if self.scale_book is not None and not self.scale_book.frozen:
            raise ValueError("scale_book must be frozen before it can quantize")
        for name in ("scale_book_kinds", "lowrank_kinds", "wide_kinds"):
            bad = set(getattr(self, name)) - {"key", "value"}
            if bad:
                raise ValueError(f"{name} must be key/value, got {sorted(bad)}")
        if self.lowrank_rank < 0:
            raise ValueError("lowrank_rank must be >= 0")
        if not 2 <= self.lowrank_factor_bits <= 16:
            raise ValueError("lowrank_factor_bits must be in [2, 16]")
        if self.refresh_every < 1 or self.min_interval < 1:
            raise ValueError("refresh intervals must be >= 1")
        if self.max_interval < self.min_interval:
            raise ValueError("max_interval must be >= min_interval")


GROUP_AXES = ("channel", "token")


def _affine_params(
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    bits: int,
    symmetric: bool,
    clip_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Scale and zero point from a range, plus the integer bounds.

    Factored out of the group-wise path so that a *static* scale and a dynamic
    one are produced by the same three lines. The whole static-scale experiment
    is a comparison between two ways of choosing ``x_min``/``x_max``, and if the
    arithmetic downstream of that choice differed even slightly the comparison
    would be measuring the difference in code.
    """
    x_max = x_max * clip_ratio
    x_min = x_min * clip_ratio

    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        qmin = -(2 ** (bits - 1))
        scale = torch.maximum(x_max.abs(), x_min.abs()) / max(qmax, 1)
        zero = torch.zeros_like(scale)
    else:
        qmin, qmax = 0, 2**bits - 1
        x_min = torch.minimum(x_min, torch.zeros_like(x_min))
        x_max = torch.maximum(x_max, torch.zeros_like(x_max))
        scale = (x_max - x_min) / (qmax - qmin)
        zero = (qmin - x_min / scale.clamp(min=1e-8)).round()

    return scale.clamp(min=1e-8), zero, qmin, qmax


def _quantize_last_axis(
    x: torch.Tensor, bits: int, group_size: int, symmetric: bool, clip_ratio: float
) -> torch.Tensor:
    """Group-wise affine fake-quantization in runs of ``group_size``, last axis.

    A length that is not a multiple of the group is padded by repeating the
    final element. Repeating a value already inside the last group cannot
    change that group's min or max, so the padding is invisible to the scales
    -- which matters because the token axis has no reason to divide evenly by
    anything, and refusing those lengths would rule out the axis entirely.
    """
    dim = x.shape[-1]
    g = min(group_size, dim)
    pad = (-dim) % g
    if pad:
        x = torch.cat([x, x[..., -1:].expand(*x.shape[:-1], pad)], dim=-1)

    orig_dtype, shape = x.dtype, x.shape
    xf = x.float().reshape(*shape[:-1], shape[-1] // g, g)

    scale, zero, qmin, qmax = _affine_params(
        xf.amin(dim=-1, keepdim=True), xf.amax(dim=-1, keepdim=True),
        bits, symmetric, clip_ratio,
    )
    q = torch.clamp(torch.round(xf / scale) + zero, qmin, qmax)
    out = ((q - zero) * scale).reshape(shape).to(orig_dtype)
    return out[..., :dim] if pad else out


def quantize_kv(
    x: torch.Tensor,
    bits: int,
    group_size: int,
    symmetric: bool = False,
    clip_ratio: float = 0.95,
    axis: str = "channel",
) -> torch.Tensor:
    """Fake-quantize a cache tensor group-wise. ``x`` is [B, heads, T, head_dim].

    ``axis`` is *which direction a group runs in*, and it is not a detail:

    * ``"channel"`` — a group is ``group_size`` neighbouring channels of one
      token of one head. One scale per token per run of channels.
    * ``"token"`` — a group is ``group_size`` neighbouring tokens of one
      channel of one head. One scale per channel per run of tokens.

    The literature names these by what a scale is shared *across*, which is the
    opposite way round, so the mapping is worth spelling out: ``"channel"``
    here is KIVI's *per-token* quantization, and ``"token"`` here is KIVI's
    *per-channel*. KIVI found the two sides of attention want different ones --
    K per-channel, V per-token -- because K's outliers live in fixed channels,
    where grouping along tokens keeps them out of everyone else's scale, while
    V's do not.

    Everything measured in this project so far used ``"channel"`` for both,
    which is the wrong half of that if it carries over to a diffusion LM. It is
    the default only so that existing numbers stay reproducible.
    """
    if bits >= 16:
        return x
    if axis not in GROUP_AXES:
        raise ValueError(f"axis must be one of {GROUP_AXES}, got {axis!r}")

    if axis == "token":
        if x.dim() < 2:
            raise ValueError("token-axis grouping needs at least [T, head_dim]")
        moved = x.transpose(-1, -2).contiguous()
        out = _quantize_last_axis(moved, bits, group_size, symmetric, clip_ratio)
        return out.transpose(-1, -2).contiguous()

    return _quantize_last_axis(x, bits, group_size, symmetric, clip_ratio)


def channel_range(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Min and max per (head, channel), over batch and every token.

    ``x`` is [B, heads, T, head_dim]; the result is two [heads, 1, head_dim]
    tensors, shaped to broadcast straight back against ``x``.

    This is the token axis taken to its limit: one group covering every token
    of a channel. That is the granularity a static scale must have, because a
    scale that lives in the model cannot depend on how many tokens are in the
    cache or where they sit.
    """
    if x.dim() != 4:
        raise ValueError(f"expected [B, heads, T, head_dim], got {tuple(x.shape)}")
    xf = x.float()
    lo = xf.amin(dim=(0, 2), keepdim=True)[0]
    hi = xf.amax(dim=(0, 2), keepdim=True)[0]
    return lo, hi


def channel_scales(
    x: torch.Tensor,
    bits: int,
    symmetric: bool = False,
    clip_ratio: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The affine scale and zero point of every channel of ``x``.

    The same numbers a token-axis dynamic quantizer would compute with a group
    spanning the whole tensor, exposed on their own because the first question
    about static scales is not how well they quantize -- it is how far they
    move between one canvas and the next. That is answerable from the scales
    alone, at the cost of one forward, and it decides whether the rest of the
    experiment is worth running.
    """
    lo, hi = channel_range(x)
    scale, zero, _, _ = _affine_params(lo, hi, bits, symmetric, clip_ratio)
    return scale, zero


def quantize_kv_static(
    x: torch.Tensor,
    bits: int,
    scale: torch.Tensor,
    zero: torch.Tensor,
    symmetric: bool = False,
) -> Tuple[torch.Tensor, float]:
    """Fake-quantize with a scale that came from somewhere else.

    ``scale``/``zero`` are [heads, 1, head_dim] and broadcast over tokens: one
    number per channel of one head, exactly what ``channel_range`` produces.
    Returns the tensor and the fraction of entries that hit an end of the
    integer range.

    That fraction is the risk a static scale carries and a dynamic one does
    not: a scale derived from the tensor it stores cannot be exceeded by it,
    while a calibrated one can, and an exceeded value is not rounded, it is
    truncated -- which is how an outlier channel loses the outlier that
    justified giving it a scale of its own.

    It is **not** zero for a perfect calibration, and reading it as if it were
    would be the same mistake as judging a control by a transplanted
    threshold. ``clip_ratio`` deliberately places the rails inside the observed
    range, so the dynamic path rails too, on any tensor with tails. The number
    means something only against that reference -- the same measurement with
    the scales taken from the tensor itself -- which is why the sweep prints
    the two side by side.
    """
    if bits >= 16:
        return x, 0.0
    qmin, qmax = (0, 2**bits - 1) if not symmetric else (
        -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    )
    xf = x.float()
    raw = torch.round(xf / scale) + zero
    q = torch.clamp(raw, qmin, qmax)
    clipped = float((raw != q).float().mean())
    return ((q - zero) * scale).to(x.dtype), clipped


def bits_per_entry(
    bits: int,
    group_size: int,
    *,
    symmetric: bool = False,
    static: bool = False,
    scale_bits: int = 16,
) -> float:
    """What one cached number really costs, scales included.

    The number a paper reports as "4-bit cache" is 4 only if the scales are
    free, and a dynamic scale is not: it is computed from the tensor being
    stored, so it has to be stored beside it. One scale and one zero in fp16
    per group is 32 bits, which is 0.25 bits per entry at a group of 128 and a
    full 1 bit at a group of 32 -- a quarter of the budget.

    A static scale lives in the model and does not scale with the cache at all
    (80 KB for this checkpoint, against 32 GB of weights), so it contributes
    nothing here. That is the entire arithmetic case for calibrating: the same
    four bits become 4.00 rather than 4.25, and 16/4.25 = 3.76x becomes 4.00x.
    """
    if bits >= 16 or static:
        return float(bits)
    per_group = scale_bits if symmetric else 2 * scale_bits
    return bits + per_group / max(group_size, 1)


def mixed_precision(
    x: torch.Tensor,
    bits: int,
    extra_bits: int,
    n_channels: int,
    group_size: int,
    axis: str = "token",
    clip_ratio: float = 0.95,
) -> torch.Tensor:
    """Give the widest ``n_channels`` channels of each head ``extra_bits`` more.

    The control a low-rank correction has to beat, and it exists because the
    residual's dominant direction turned out to be a single *channel* at every
    layer and every mask ratio -- 1.0 to 1.7 channels carrying it, against 105
    to 154 tokens. A rank-1 term is an outer product, so it pays for T + D
    numbers and their scales to encode something that is one channel deep.
    Spending the same budget directly on that channel needs no factors, no
    scales for the factors, and no matrix product at read time.

    Which channels are widest is a property of the layer, not of the canvas --
    that is the same observation the token axis won on -- so the index list is
    model metadata and costs the cache nothing, exactly like a static scale.
    """
    q = quantize_kv(x, bits, group_size, clip_ratio=clip_ratio, axis=axis)
    if n_channels <= 0 or extra_bits <= 0:
        return q
    fine = quantize_kv(x, bits + extra_bits, group_size, clip_ratio=clip_ratio,
                       axis=axis)

    lo, hi = channel_range(x)                       # [heads, 1, head_dim]
    width = (hi - lo)[:, 0, :]                      # [heads, head_dim]
    k = min(n_channels, width.shape[-1])
    picked = torch.zeros_like(width, dtype=torch.bool)
    picked.scatter_(-1, width.topk(k, dim=-1).indices, True)
    return torch.where(picked[None, :, None, :].expand_as(x), fine, q)


def mixed_precision_bits(
    bits: int, extra_bits: int, n_channels: int, head_dim: int
) -> float:
    """Bits per entry for the widened channels, over the whole tensor.

    The channel indices are not charged for: they are fixed per layer and live
    in the model, like a static scale. The scale count does not change either
    -- the same groups, wider values in a few of them.
    """
    return bits + extra_bits * min(n_channels, head_dim) / head_dim


def lowrank_residual(
    x: torch.Tensor,
    xq: torch.Tensor,
    rank: int,
    factor_bits: int = 16,
    factor_scales: str = "column",
) -> Tuple[torch.Tensor, float]:
    """The best rank-``r`` approximation of what quantization threw away.

    ``K ~ Q(K) + A.B``, per head: the residual of one head is a [T, head_dim]
    matrix and its truncated SVD is the optimal correction at that rank. The
    idea is established on *weights* (LQER, CALDERA, ZeroQuant-V2) and untried
    on a diffusion LM's cache, DART included.

    Returns the correction and the fraction of residual energy it captures --
    the second is the number that decides whether any of this is worth its
    bits, and it is cheap to obtain, so it is returned from the same call
    rather than left to a separate pass.

    Whether the residual *has* a low-rank structure is the whole question, and
    the obvious answer is wrong in an instructive way. Rounding noise has no
    preferred directions, so rank r of head_dim should capture about r/head_dim
    of it -- 6% at rank 8 of 128. It captures far more, because grouping along
    tokens gives every channel its own scale and a channel four times wider
    rounds four times coarser: the residual inherits the channel structure of
    the tensor, and a few outlier channels carry most of its energy. On
    Gaussian channels with the gains this model's outlier survey reports, rank
    4 recovers 56% of it.

    Which does not make it a good deal, and the arithmetic says so before any
    measurement: recovering 56% leaves the residual at 0.66x for 0.79 bits,
    where one more flat bit leaves it at 0.47x. The correction is competing
    against the cheapest possible use of the same budget, and against a scheme
    that spends the budget where the correction found the energy -- the fat
    channels -- which is mixed precision, not low rank.

    ``factor_bits`` is not decoration. The plan's arithmetic -- 2816 numbers
    against 28672, "10% overhead, 3 bits + rank 8 = 3.3 bits effective" --
    counts *numbers*, and the numbers are fp16 while the entries are three
    bits. In bits the same correction costs 1.57 per entry, so "3 bits + rank
    8" is 4.57 effective and loses to four bits flat before it is measured. It
    reaches 3.3 only if the factors are themselves stored at four bits, which
    is a choice with its own error and is therefore made explicit here.
    """
    if rank <= 0:
        return torch.zeros_like(x), 0.0
    if x.dim() != 4:
        raise ValueError(f"expected [B, heads, T, head_dim], got {tuple(x.shape)}")

    resid = (x.float() - xq.float())
    r = min(rank, resid.shape[-1], resid.shape[-2])
    u, s, vh = torch.linalg.svd(resid, full_matrices=False)

    total = s.pow(2).sum(dim=-1).clamp(min=1e-12)
    captured = float((s[..., :r].pow(2).sum(dim=-1) / total).mean())

    a = u[..., :r] * s[..., :r].unsqueeze(-2)        # [B, heads, T, r]
    b = vh[..., :r, :]                               # [B, heads, r, head_dim]
    if factor_bits < 16:
        # How finely A's own scales are cut is not a detail at low rank. One
        # scale per token is T of them, which at rank 1 costs five times more
        # than the factor it describes -- the accounting was measuring the
        # scales and calling it the correction. One scale per column is r of
        # them, coarser and nearly free. B is r scales either way.
        if factor_scales == "token":
            a = _quantize_last_axis(a, factor_bits, a.shape[-1], False, 1.0)
        elif factor_scales == "column":
            a = _quantize_last_axis(
                a.transpose(-1, -2).contiguous(), factor_bits,
                a.shape[-2], False, 1.0,
            ).transpose(-1, -2)
        else:
            raise ValueError("factor_scales must be 'token' or 'column'")
        b = _quantize_last_axis(b, factor_bits, b.shape[-1], False, 1.0)
    return (a @ b).to(x.dtype), captured


def lowrank_bits(
    rank: int,
    n_tokens: int,
    head_dim: int,
    factor_bits: int = 16,
    scale_bits: int = 16,
    factor_scales: str = "column",
) -> float:
    """Bits per stored entry that a rank-``rank`` correction adds.

    Counts the factors and, when they are quantized, their own scales -- a
    correction that needs scales to be read back is not free of them either,
    and at rank 1 with a scale per token they are five times the factor.
    """
    if rank <= 0:
        return 0.0
    r = min(rank, head_dim, n_tokens)
    values = r * (n_tokens + head_dim) * factor_bits
    if factor_bits >= 16:
        scales = 0
    elif factor_scales == "token":
        scales = 2 * scale_bits * (n_tokens + r)
    else:
        scales = 2 * scale_bits * 2 * r
    return (values + scales) / (n_tokens * head_dim)


@dataclass
class StaticScaleBook:
    """K/V scales calibrated once and kept in the model, bucketed by mask ratio.

    Two measured facts put this here rather than in a list of ideas.

    *The token axis won because K's outliers sit in fixed channels* -- four bits
    along tokens were indistinguishable from an exact cache, where the same four
    along channels were not. A quantity that sits in a fixed channel is exactly
    the quantity that can be measured once and reused, so the axis result and
    this one hold each other up.

    *The distribution moves with the mask ratio* -- outliers 4.1 at full mask
    against 8.7 on a decoded canvas, decision margin 1.988 against 0.756. So one
    static scale for the whole trajectory is the wrong object, and the fix is
    free: the mask ratio is known at every sampler step at no cost, so the
    scales can be bucketed by it and selected by the current one.

    Calibration is a range, not a fit: ``observe`` accumulates per-channel
    extremes and ``freeze`` turns them into the same affine scale a dynamic
    quantizer would have produced. ``reduce`` decides what "the" range of a
    channel is across canvases -- ``"max"`` takes the envelope, which never
    clips and buys that with a coarser step; ``"mean"`` takes the average
    canvas, which is finer everywhere and clips on the tails. Which is better is
    the measurement, not an assumption, so both are here.
    """

    bits: int = 4
    symmetric: bool = False
    clip_ratio: float = 0.95
    # Representative mask ratios. A lookup snaps to the nearest one, so a
    # trajectory never lands outside the book. Five is the resolution the
    # outlier sweep was taken at; one bucket is the ablation that says whether
    # bucketing was needed at all.
    buckets: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    reduce: str = "max"

    def __post_init__(self) -> None:
        if self.reduce not in ("max", "mean"):
            raise ValueError("reduce must be 'max' or 'mean'")
        if not self.buckets:
            raise ValueError("need at least one bucket")
        self.buckets = tuple(sorted(float(b) for b in self.buckets))
        # key -> [lo, hi, n]; key is (bucket, layer, kind)
        self._acc: Dict[Tuple[float, int, str], List] = {}
        self._params: Dict[Tuple[float, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}
        self.frozen = False
        # How often a lookup had to fall back to a neighbouring bucket. Counted
        # rather than silently allowed: a fallback that nobody notices is how
        # the CGQ artifact came to be signed with a component it never ran.
        self.fallbacks = 0

    # ------------------------------------------------------------ calibration

    def bucket_for(self, mask_ratio: float) -> float:
        return min(self.buckets, key=lambda b: abs(b - float(mask_ratio)))

    def observe(self, layer: int, kind: str, x: torch.Tensor, mask_ratio: float) -> None:
        """Add one canvas's K or V to the calibration set."""
        if self.frozen:
            raise RuntimeError("book is frozen; build a new one to recalibrate")
        if kind not in ("key", "value"):
            raise ValueError(f"kind must be 'key' or 'value', got {kind!r}")
        lo, hi = channel_range(x)
        key = (self.bucket_for(mask_ratio), int(layer), kind)
        acc = self._acc.get(key)
        if acc is None:
            self._acc[key] = [lo.cpu(), hi.cpu(), 1]
            return
        lo, hi = lo.cpu(), hi.cpu()
        if self.reduce == "max":
            acc[0] = torch.minimum(acc[0], lo)
            acc[1] = torch.maximum(acc[1], hi)
        else:
            acc[0] = acc[0] + lo
            acc[1] = acc[1] + hi
        acc[2] += 1

    def freeze(self) -> "StaticScaleBook":
        for key, (lo, hi, n) in self._acc.items():
            if self.reduce == "mean":
                lo, hi = lo / n, hi / n
            scale, zero, _, _ = _affine_params(
                lo, hi, self.bits, self.symmetric, self.clip_ratio
            )
            self._params[key] = (scale, zero)
        if not self._params:
            raise RuntimeError("nothing was observed: the book would quantize nothing")
        self.frozen = True
        return self

    # ---------------------------------------------------------------- lookup

    def get(
        self, layer: int, kind: str, mask_ratio: float, device=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.frozen:
            raise RuntimeError("call freeze() before using the book")
        bucket = self.bucket_for(mask_ratio)
        key = (bucket, int(layer), kind)
        if key not in self._params:
            candidates = [b for (b, ell, k) in self._params if ell == layer and k == kind]
            if not candidates:
                raise KeyError(f"nothing calibrated for layer {layer}, {kind}")
            self.fallbacks += 1
            key = (min(candidates, key=lambda b: abs(b - bucket)), int(layer), kind)
        scale, zero = self._params[key]
        if device is not None:
            scale, zero = scale.to(device), zero.to(device)
        return scale, zero

    # -------------------------------------------------------------- accounting

    @property
    def n_scalars(self) -> int:
        """Scales plus zero points held, i.e. what the model has to carry."""
        return 2 * sum(int(s.numel()) for s, _ in self._params.values())

    def describe(self) -> str:
        buckets = sorted({b for b, _, _ in self._params})
        layers = len({ell for _, ell, _ in self._params})
        kb = self.n_scalars * 2 / 1024        # fp16 storage
        return (
            f"static scales: {self.bits} bits, reduce={self.reduce}, "
            f"{len(buckets)} bucket(s) {buckets}, {layers} layers, "
            f"{self.n_scalars} numbers = {kb:.0f} KB in fp16"
        )


@dataclass
class CacheStats:
    """What has to be recorded to tell the two error sources apart."""

    refreshes: int = 0
    reuses: int = 0
    entries_written: int = 0
    # The current block's store, counted apart from the prefix's. Pooling them
    # would hide the quantity the refresh policy is tuned against: the prefix
    # is rewritten once per block whatever the policy says, so a pooled hit
    # rate mostly reports the block length.
    window_refreshes: int = 0
    window_reuses: int = 0
    window_ages: List[int] = field(default_factory=list)
    # Age (in denoising steps) of every entry at the moment it was read.
    ages: List[int] = field(default_factory=list)
    # Steps a confidence-thresholded sampler actually needed, one entry per
    # block. Under a fixed schedule this stays empty: there the number of
    # steps is an input, not an observation, and the parallelism it implies
    # cannot depend on the cache. It is the only quantity in this file that
    # the cache is allowed to make worse in a way a user would feel -- the
    # decisions it damages turn out not to change the answer, so throughput
    # is where the price lands.
    steps_used: List[int] = field(default_factory=list)
    # Relative error between the cached tensor and a freshly computed one,
    # measured whenever ``measure_drift`` is called.
    staleness_error: List[float] = field(default_factory=list)
    quantization_error: List[float] = field(default_factory=list)
    # Fraction of entries a static scale railed, one entry per write. Empty
    # under dynamic scales. Not comparable to zero -- clip_ratio rails a few
    # percent even when the calibration is exact -- only to the same figure
    # with the scales taken from the stored tensor itself.
    clipped: List[float] = field(default_factory=list)
    # Share of the quantization residual a low-rank correction recovered, one
    # entry per corrected write. The quantity that decides whether the rank is
    # worth the bits it costs.
    lowrank_captured: List[float] = field(default_factory=list)
    # Which layer each of those came from. Kept because the two errors are
    # expected to behave differently with depth and the averages hide it:
    # a layer's K/V are computed from the previous layer's already-stale
    # output, so staleness should compound down the stack, while rounding is
    # redrawn at every write and should not.
    drift_layers: List[int] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        total = self.refreshes + self.reuses
        return self.reuses / total if total else 0.0

    @property
    def mean_age(self) -> float:
        return sum(self.ages) / len(self.ages) if self.ages else 0.0

    @property
    def window_hit_rate(self) -> float:
        """Share of steps that reused the current block instead of recomputing.

        This is the number the whole current-block experiment turns on: it is
        what is saved, and staleness is what it costs.
        """
        total = self.window_refreshes + self.window_reuses
        return self.window_reuses / total if total else 0.0

    @property
    def mean_window_age(self) -> float:
        return (sum(self.window_ages) / len(self.window_ages)
                if self.window_ages else 0.0)

    @property
    def clip_rate(self) -> float:
        """Mean share of entries clipped by a static scale, over all writes."""
        return sum(self.clipped) / len(self.clipped) if self.clipped else 0.0

    def drift_by_layer(self) -> Dict[int, Dict[str, float]]:
        """Mean staleness and rounding per layer, and how many samples each.

        The aggregate over all layers answers "how wrong is the cache"; this
        answers "where", which is the question a refresh policy is tuned
        against -- refreshing every layer at the same interval is only right
        if the error is flat across depth.
        """
        out: Dict[int, Dict[str, float]] = {}
        for layer, stale, quant in zip(
            self.drift_layers, self.staleness_error, self.quantization_error
        ):
            acc = out.setdefault(layer, {"staleness": 0.0, "quantization": 0.0, "n": 0})
            acc["staleness"] += stale
            acc["quantization"] += quant
            acc["n"] += 1
        for acc in out.values():
            n = max(acc["n"], 1)
            acc["staleness"] /= n
            acc["quantization"] /= n
        return out

    def summary(self) -> str:
        lines = [
            f"cache: {self.refreshes} refreshes, {self.reuses} reuses "
            f"(hit rate {100 * self.hit_rate:.1f}%), mean age "
            f"{self.mean_age:.2f} steps"
        ]
        if self.staleness_error:
            s = sum(self.staleness_error) / len(self.staleness_error)
            lines.append(f"  staleness error   {s:.5f}")
        if self.quantization_error:
            q = sum(self.quantization_error) / len(self.quantization_error)
            lines.append(f"  quantization error {q:.5f}")
        if self.lowrank_captured:
            c = sum(self.lowrank_captured) / len(self.lowrank_captured)
            lines.append(f"  low rank recovered {100 * c:.1f}% of the residual")
        if self.clipped:
            lines.append(
                f"  clipped by static scales: {100 * self.clip_rate:.3f}% of entries"
            )
        if self.staleness_error and self.quantization_error:
            s = sum(self.staleness_error) / len(self.staleness_error)
            q = sum(self.quantization_error) / len(self.quantization_error)
            if q > 0:
                lines.append(f"  ratio staleness/quantization: {s / q:.2f}")

        by_layer = self.drift_by_layer()
        if len(by_layer) > 1:
            # First and last, because the question is whether it compounds.
            layers = sorted(by_layer)
            first, last = by_layer[layers[0]], by_layer[layers[-1]]
            lines.append(
                f"  by depth: layer {layers[0]} staleness {first['staleness']:.5f} "
                f"/ rounding {first['quantization']:.5f}"
            )
            lines.append(
                f"            layer {layers[-1]} staleness {last['staleness']:.5f} "
                f"/ rounding {last['quantization']:.5f}"
            )
        return "\n".join(lines)


class BlockKVCache:
    """Per-layer K/V store with a refresh policy and quantization at write.

    Deliberately dumb about attention: it stores tensors and answers "should I
    recompute this step?".  Wiring it into a model's attention is the
    model-specific part.
    """

    def __init__(self, cfg: KVCacheConfig, n_layers: int):
        self.cfg = cfg
        self.n_layers = n_layers
        self.stats = CacheStats()
        self._k: Dict[int, torch.Tensor] = {}
        self._v: Dict[int, torch.Tensor] = {}
        self._written_at: Dict[int, int] = {}
        # The block being decoded right now, stored separately from the prefix.
        # The two are different objects: a closed block cannot change, so the
        # prefix is exact and only rounding is at stake, while the current
        # block's tokens are committed between steps and its K/V go stale the
        # moment they do. Keeping them apart is what lets the two errors be
        # measured apart -- which is the whole reason this model was chosen.
        self._wk: Dict[int, torch.Tensor] = {}
        self._wv: Dict[int, torch.Tensor] = {}
        self._window_written_at: Dict[int, int] = {}
        self.step = 0
        # Mask ratio of what is being stored, used to pick a bucket of static
        # scales. A caller that passes ``mask`` to ``write`` need not set it --
        # the ratio is read off the mask itself, which is the point: in a real
        # sampler this quantity is already known at every step and costs
        # nothing to obtain.
        self.mask_ratio: Optional[float] = None

    # ------------------------------------------------------------- policy

    def should_refresh(
        self,
        layer: int,
        step: int,
        mask_ratio: float = 1.0,
        block_boundary: bool = False,
    ) -> bool:
        """Whether layer ``layer`` must recompute K/V at this step."""
        if layer not in self._k:
            return True  # nothing cached yet

        p = self.cfg.policy
        if p == "never":
            return block_boundary
        if p == "block":
            return block_boundary
        if p == "every_n":
            return block_boundary or (step - self._written_at[layer]) >= self.cfg.refresh_every
        if p == "mask_ratio":
            return block_boundary or (step - self._written_at[layer]) >= self.interval_for(mask_ratio)
        raise AssertionError(f"unhandled policy {p}")

    def interval_for(self, mask_ratio: float) -> int:
        """Refresh interval under the adaptive policy.

        Everything moves while the sequence is mostly masked, so the cache goes
        stale fast there; late in the trajectory most tokens are fixed and it
        barely ages.  Mask ratio is known for free at every step.
        """
        lo, hi = self.cfg.min_interval, self.cfg.max_interval
        ratio = min(max(mask_ratio, 0.0), 1.0)
        return max(1, int(round(lo + (hi - lo) * (1.0 - ratio))))

    # -------------------------------------------------------------- store

    def write(
        self,
        layer: int,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize and store; returns what a reader would get back.

        ``mask`` is [B, T] and True where the position is still masked; when
        the two bit widths differ, masked and decoded positions are quantized
        separately.
        """
        ratio = self.mask_ratio
        if mask is not None:
            ratio = float(mask.to(torch.float32).mean())
        kq = self._correct(k, self._quantize_with_mask(k, mask, "key", layer, ratio),
                           "key")
        vq = self._correct(v, self._quantize_with_mask(v, mask, "value", layer, ratio),
                           "value")

        self._k[layer], self._v[layer] = kq, vq
        self._written_at[layer] = self.step
        self.stats.refreshes += 1
        self.stats.entries_written += int(k.shape[-2]) if k.dim() >= 2 else 0
        return kq, vq

    def _correct(self, x: torch.Tensor, xq: torch.Tensor, kind: str) -> torch.Tensor:
        """Add the low-rank correction, if this side was given one."""
        cfg = self.cfg
        if cfg.lowrank_rank <= 0 or kind not in cfg.lowrank_kinds:
            return xq
        if self._bits_for(kind, "decoded") >= 16:
            return xq          # nothing was thrown away, so nothing to recover
        correction, captured = lowrank_residual(
            x, xq, cfg.lowrank_rank, cfg.lowrank_factor_bits
        )
        self.stats.lowrank_captured.append(captured)
        return xq + correction

    def _bits_for(self, kind: str, status: str) -> int:
        """Bit width for one side of the cache at one position status.

        ``key_bits`` / ``value_bits`` override the status-based widths when
        set: the question "does K need more precision than V" is separate from
        "does a masked position need less than a decoded one", and answering
        one must not silently answer the other.
        """
        override = self.cfg.key_bits if kind == "key" else self.cfg.value_bits
        if override is not None:
            return override
        return self.cfg.decoded_bits if status == "decoded" else self.cfg.masked_bits

    def _axis_for(self, kind: str) -> str:
        return self.cfg.key_axis if kind == "key" else self.cfg.value_axis

    def _quantize_static(
        self,
        x: torch.Tensor,
        kind: str,
        layer: int,
        mask_ratio: Optional[float],
        record: bool = True,
    ) -> torch.Tensor:
        """Store one side of the cache with scales that came from calibration.

        Refuses rather than falls back when the widths disagree. A scale is
        computed for a particular number of levels, so a book calibrated at
        four bits used at three is not a slightly worse quantizer, it is the
        wrong one -- and it would still produce a table.
        """
        book = self.cfg.scale_book
        bits = self._bits_for(kind, "decoded")
        if bits != book.bits:
            raise ValueError(
                f"{kind} is configured for {bits} bits but the scale book was "
                f"calibrated at {book.bits}"
            )
        if mask_ratio is None:
            raise ValueError(
                "static scales are selected by mask ratio: pass `mask` to "
                "write() or set cache.mask_ratio"
            )
        scale, zero = book.get(layer, kind, mask_ratio, device=x.device)
        out, clipped = quantize_kv_static(x, bits, scale, zero, book.symmetric)
        if record:
            self.stats.clipped.append(clipped)
        return out

    def _quantize_with_mask(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        kind: str = "key",
        layer: int = 0,
        mask_ratio: Optional[float] = None,
        record: bool = True,
    ) -> torch.Tensor:
        cfg = self.cfg
        decoded_bits = self._bits_for(kind, "decoded")
        masked_bits = self._bits_for(kind, "masked")
        axis = self._axis_for(kind)

        if cfg.scale_book is not None and kind in cfg.scale_book_kinds:
            # A static scale is per-channel by construction, so the two
            # status-dependent widths have nowhere to live here: a group spans
            # every token of a channel, masked and decoded alike.
            if decoded_bits != masked_bits:
                raise ValueError(
                    "static scales cannot carry two bit widths: a calibrated "
                    "group spans decoded and masked positions together"
                )
            return self._quantize_static(x, kind, layer, mask_ratio, record)

        widened = cfg.wide_channels > 0 and cfg.wide_extra_bits > 0 \
            and kind in cfg.wide_kinds
        if widened:
            # A width split is per channel, so it has the same trouble with the
            # status split that a static scale does: a group runs across
            # masked and decoded positions together. Refused rather than
            # silently resolved either way.
            if decoded_bits != masked_bits:
                raise ValueError(
                    "widened channels cannot carry two bit widths: the split "
                    "is per channel and a group spans both statuses"
                )
            return mixed_precision(
                x, decoded_bits, cfg.wide_extra_bits, cfg.wide_channels,
                cfg.group_size, axis=axis, clip_ratio=cfg.clip_ratio,
            )

        if mask is None or decoded_bits == masked_bits:
            return quantize_kv(
                x, decoded_bits, cfg.group_size, cfg.symmetric, cfg.clip_ratio,
                axis=axis,
            )

        # x is [B, heads, T, head_dim]; mask is [B, T].
        if x.dim() != 4:
            raise ValueError(f"expected [B, heads, T, head_dim], got {tuple(x.shape)}")
        m = mask.to(torch.bool)
        while m.dim() < x.dim() - 1:
            m = m.unsqueeze(1)
        m = m.unsqueeze(-1).expand_as(x)

        masked = quantize_kv(
            x, masked_bits, cfg.group_size, cfg.symmetric, cfg.clip_ratio, axis=axis
        )
        decoded = quantize_kv(
            x, decoded_bits, cfg.group_size, cfg.symmetric, cfg.clip_ratio, axis=axis
        )
        return torch.where(m, masked, decoded)

    def scramble(self, generator: Optional[torch.Generator] = None) -> None:
        """Permute stored entries along the token axis: the chance floor.

        A control, not a feature. Bit-width sweeps report an error and an
        argmax agreement, and neither has a meaningful zero: it is not obvious
        from the numbers alone whether "argmax kept 53%" means half the
        decisions survived or the cache is dead and 53% is what agreement looks
        like when the two runs share nothing but the model's priors. Measuring
        with a cache whose information is destroyed but whose distribution is
        untouched gives that floor, and any bit width sitting at it is carrying
        nothing, whatever its error says.

        Permutation rather than noise or zeros on purpose: it preserves the
        exact marginal distribution of every channel, so the floor is not
        confounded by a scale the model never sees in practice.

        A fresh permutation per layer and per side, deliberately. One shared
        permutation would hand the model a shuffled but internally consistent
        sequence -- keys still matched to their own values, every layer
        agreeing on where each position went -- and that is a bag of words, not
        nothing. Drawing independently breaks the K/V pairing and the agreement
        across depth, which is what makes this a floor rather than a weaker
        cache.
        """
        for store in (self._k, self._v, self._wk, self._wv):
            for layer, t in list(store.items()):
                idx = torch.randperm(t.shape[-2], generator=generator)
                store[layer] = t.index_select(-2, idx.to(t.device))

    # ------------------------------------------------------- current block

    def should_refresh_window(
        self, layer: int, step: int, mask_ratio: float = 1.0
    ) -> bool:
        """Whether the block being decoded must recompute its own K/V.

        Separate from ``should_refresh`` because the two stores answer to
        different clocks: the prefix is rewritten once per block boundary and
        is exact in between, while the current block ages every time a token in
        it commits. A single interval for both would tie the cheap decision to
        the expensive one.
        """
        if layer not in self._wk:
            return True
        p = self.cfg.policy
        if p in ("never", "block"):
            return False
        age = step - self._window_written_at[layer]
        if p == "every_n":
            return age >= self.cfg.refresh_every
        if p == "mask_ratio":
            return age >= self.interval_for(mask_ratio)
        raise AssertionError(f"unhandled policy {p}")

    def write_window(
        self,
        layer: int,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize and store the current block's K/V; returns what is stored."""
        ratio = self.mask_ratio
        if mask is not None:
            ratio = float(mask.to(torch.float32).mean())
        kq = self._correct(k, self._quantize_with_mask(k, mask, "key", layer, ratio),
                           "key")
        vq = self._correct(v, self._quantize_with_mask(v, mask, "value", layer, ratio),
                           "value")
        self._wk[layer], self._wv[layer] = kq, vq
        self._window_written_at[layer] = self.step
        self.stats.window_refreshes += 1
        return kq, vq

    def read_window(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer not in self._wk:
            raise KeyError(f"layer {layer} has no current block cached")
        self.stats.window_reuses += 1
        self.stats.window_ages.append(self.step - self._window_written_at[layer])
        return self._wk[layer], self._wv[layer]

    def has_window(self, layer: int) -> bool:
        return layer in self._wk

    def window_age(self, layer: int) -> int:
        return self.step - self._window_written_at.get(layer, self.step)

    def reset_window(self) -> None:
        """Drop the current block's entries. Called when the block moves on.

        Not optional: the store is indexed by layer, not by position, so
        carrying it across a block boundary would hand the next block the
        previous one's keys at the previous one's positions.
        """
        self._wk.clear()
        self._wv.clear()
        self._window_written_at.clear()

    def read(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer not in self._k:
            raise KeyError(f"layer {layer} has nothing cached")
        self.stats.reuses += 1
        self.stats.ages.append(self.step - self._written_at[layer])
        return self._k[layer], self._v[layer]

    def has(self, layer: int) -> bool:
        return layer in self._k

    def age(self, layer: int) -> int:
        return self.step - self._written_at.get(layer, self.step)

    def advance(self) -> None:
        self.step += 1

    def reset(self) -> None:
        self._k.clear()
        self._v.clear()
        self._written_at.clear()
        self.reset_window()
        self.step = 0

    # ----------------------------------------------------------- analysis

    def measure_drift(
        self, layer: int, k_fresh: torch.Tensor, v_fresh: torch.Tensor
    ) -> Dict[str, float]:
        """Split the cache's total error into staleness and rounding.

        Call with the K/V that a full recompute *would* have produced.  The
        cached tensor differs from it for two reasons at once; quantizing the
        fresh tensor separately isolates the rounding part.
        """
        k_cached, v_cached = self._k[layer], self._v[layer]
        # The same code path the write took, so that the isolated rounding term
        # is this cache's rounding and not a dynamic stand-in for it -- under a
        # static book those are different quantizers. ``record=False``: this is
        # a measurement, and it must not enter the write statistics.
        k_fresh_q = self._quantize_with_mask(
            k_fresh, None, "key", layer, self.mask_ratio, record=False
        )

        def rel(a: torch.Tensor, b: torch.Tensor) -> float:
            denom = b.float().pow(2).mean().sqrt().clamp(min=1e-8)
            return float((a.float() - b.float()).pow(2).mean().sqrt() / denom)

        total = rel(k_cached, k_fresh)
        quant = rel(k_fresh_q, k_fresh)
        # What the cache would have cost with no quantization at all.
        stale = rel(k_cached, k_fresh_q)

        self.stats.staleness_error.append(stale)
        self.stats.quantization_error.append(quant)
        self.stats.drift_layers.append(layer)
        return {
            "total": total, "staleness": stale, "quantization": quant,
            "age": float(self.age(layer)),
        }


__all__ = [
    "KVCacheConfig",
    "BlockKVCache",
    "CacheStats",
    "quantize_kv",
    "quantize_kv_static",
    "channel_range",
    "channel_scales",
    "bits_per_entry",
    "lowrank_residual",
    "lowrank_bits",
    "mixed_precision",
    "mixed_precision_bits",
    "StaticScaleBook",
]
