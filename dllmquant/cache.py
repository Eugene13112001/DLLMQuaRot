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
    # Entries at masked positions are wrong the moment their token changes.
    # Refreshing only those costs a fraction of a full refresh.
    refresh_masked_only: bool = False

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
        if self.refresh_every < 1 or self.min_interval < 1:
            raise ValueError("refresh intervals must be >= 1")
        if self.max_interval < self.min_interval:
            raise ValueError("max_interval must be >= min_interval")


GROUP_AXES = ("channel", "token")


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

    x_max = xf.amax(dim=-1, keepdim=True) * clip_ratio
    x_min = xf.amin(dim=-1, keepdim=True) * clip_ratio

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

    scale = scale.clamp(min=1e-8)
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


@dataclass
class CacheStats:
    """What has to be recorded to tell the two error sources apart."""

    refreshes: int = 0
    reuses: int = 0
    entries_written: int = 0
    # Age (in denoising steps) of every entry at the moment it was read.
    ages: List[int] = field(default_factory=list)
    # Relative error between the cached tensor and a freshly computed one,
    # measured whenever ``measure_drift`` is called.
    staleness_error: List[float] = field(default_factory=list)
    quantization_error: List[float] = field(default_factory=list)
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
        self.step = 0

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
        kq = self._quantize_with_mask(k, mask, "key")
        vq = self._quantize_with_mask(v, mask, "value")

        self._k[layer], self._v[layer] = kq, vq
        self._written_at[layer] = self.step
        self.stats.refreshes += 1
        self.stats.entries_written += int(k.shape[-2]) if k.dim() >= 2 else 0
        return kq, vq

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

    def _quantize_with_mask(
        self, x: torch.Tensor, mask: Optional[torch.Tensor], kind: str = "key"
    ) -> torch.Tensor:
        cfg = self.cfg
        decoded_bits = self._bits_for(kind, "decoded")
        masked_bits = self._bits_for(kind, "masked")
        axis = self._axis_for(kind)

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
        for store in (self._k, self._v):
            for layer, t in list(store.items()):
                idx = torch.randperm(t.shape[-2], generator=generator)
                store[layer] = t.index_select(-2, idx.to(t.device))

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
        k_fresh_q = quantize_kv(
            k_fresh, self._bits_for("key", "decoded"), self.cfg.group_size,
            self.cfg.symmetric, self.cfg.clip_ratio, axis=self._axis_for("key"),
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


__all__ = ["KVCacheConfig", "BlockKVCache", "CacheStats", "quantize_kv"]
