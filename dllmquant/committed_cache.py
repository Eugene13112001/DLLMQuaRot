"""A cache store that rounds a block when it is committed, and nothing else.

Block dLLMs of the SDAR / Fast-dLLM-v2 kind keep two kinds of K/V: the prompt and the
finished blocks, written once into a ``DynamicCache`` and read by every later block; and
the block being decoded, recomputed at every step and never written. Only the first kind is
compressed -- it is the bulk of the memory, and the current block is where the model makes
its decisions. So the quantizer sits in ``update``: whatever reaches the store is rounded on
the way in, and the current block, which never reaches it, cannot be rounded by accident.
The stats record every write, so a model that does write its current block (and crops it
later) shows up as the wrong write pattern instead of as a quietly different experiment.

The rounding is fake: quantize and dequantize at once, so the store holds exactly the error
a packed store would hold. That is enough for accuracy; memory and speed are not measured
here.

Names follow the literature, not ``quantize_kv``, because the opposite convention there has
already cost a misread table:

* ``per_channel`` keys -- KIVI: a group is ``group`` neighbouring tokens of one channel, one
  scale per channel per group. Outlier channels stay out of everyone else's range.
* ``per_token`` keys or values -- a group is ``group`` neighbouring channels of one token, one
  scale per token per group. The attention-bias correction needs this form: every token
  has its own step.

Per-channel groups run across tokens, so at block 4 a group of 32 is only filled after eight
blocks. ``residual`` says what happens meanwhile, and the two answers cost differently:

* ``"buffer"`` (KIVI's own) -- tokens stay exact until a group is full, then the group is
  rounded. The exact tail is real memory and is counted as such.
* ``"per_write"`` -- every write is rounded on its own, the last group shorter. No exact
  tail, but a block of 4 pays a full scale and zero for four tokens.

Neither is free, which is why ``honest_bits`` exists: bits per stored number with the scales,
zeros and exact remainders all included, so block sizes are compared at what they really cost.

Sink protection keeps chosen absolute positions exact (SDAR's sink is the third token of the
text, position 2). Inside a per-channel group the sink is first replaced by its neighbour, so
its outlier does not set the range of the tokens beside it, and then written back exact.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .cache import quantize_kv

KEY_SCALES = ("per_channel", "per_token")
RESIDUALS = ("buffer", "per_write")

# quantize_kv names the grouping direction; these name what a scale is shared across.
_AXIS = {"per_channel": "token", "per_token": "channel"}


@dataclass(frozen=True)
class CommittedCacheConfig:
    key_bits: int = 16
    value_bits: int = 16
    key_scale: str = "per_channel"
    group: int = 32
    residual: str = "buffer"
    exact_positions: Tuple[int, ...] = ()
    clip_ratio: float = 1.0   # plain min-max, as KIVI and BitSieve
    scale_bits: int = 16      # one fp16 scale and one fp16 zero per group

    def __post_init__(self):
        if self.key_scale not in KEY_SCALES:
            raise ValueError(f"key_scale must be one of {KEY_SCALES}, got {self.key_scale!r}")
        if self.residual not in RESIDUALS:
            raise ValueError(f"residual must be one of {RESIDUALS}, got {self.residual!r}")
        if self.group < 1:
            raise ValueError("group must be positive")

    def describe(self) -> str:
        sink = f", exact at {list(self.exact_positions)}" if self.exact_positions else ""
        res = f", residual {self.residual}" if self.key_scale == "per_channel" else ""
        return (f"K {self.key_bits} bits {self.key_scale}{res}, V {self.value_bits} bits "
                f"per_token, group {self.group}{sink}")


@dataclass
class _Side:
    """Counts for one of K or V, summed over layers, in tokens and groups."""
    tokens: int = 0          # stored
    quantized: int = 0       # rounded and not written back exact
    pairs: int = 0           # scale/zero pairs stored
    err: float = 0.0         # sum of squared rounding error
    norm: float = 0.0        # sum of squares of what was rounded


@dataclass
class CommittedCacheStats:
    write_lengths: List[int] = field(default_factory=list)   # layer 0, in order
    crops: int = 0
    k: _Side = field(default_factory=_Side)
    v: _Side = field(default_factory=_Side)
    numbers_per_token: int = 0    # heads x head_dim, the same for K and V

    @property
    def writes(self) -> int:
        return len(self.write_lengths)

    @property
    def stored_tokens(self) -> int:
        return sum(self.write_lengths)

    def key_rel_error(self) -> float:
        """Relative squared error of the rounded keys: grows as the width shrinks."""
        return self.k.err / self.k.norm if self.k.norm else 0.0

    def value_rel_error(self) -> float:
        return self.v.err / self.v.norm if self.v.norm else 0.0

    def honest_bits(self, cfg: CommittedCacheConfig) -> Dict[str, float]:
        """Bits per stored number, scales and exact entries included. K, V and both."""
        out = {}
        for name, side, bits in (("k", self.k, cfg.key_bits), ("v", self.v, cfg.value_bits)):
            if side.tokens == 0:
                out[name] = float(bits if bits < 16 else 16)
                continue
            if bits >= 16:
                out[name] = 16.0
                continue
            exact = side.tokens - side.quantized
            total = side.tokens * self.numbers_per_token
            cost = (side.quantized * self.numbers_per_token * bits
                    + exact * self.numbers_per_token * 16
                    + side.pairs * 2 * cfg.scale_bits)
            out[name] = cost / total
        out["kv"] = (out["k"] + out["v"]) / 2
        return out

    def describe(self, cfg: CommittedCacheConfig) -> str:
        hb = self.honest_bits(cfg)
        return (f"committed cache: {self.writes} writes, {self.stored_tokens} tokens, "
                f"{self.crops} crops; honest bits K {hb['k']:.2f} V {hb['v']:.2f}; "
                f"key rel. error {self.key_rel_error():.4f}")


def _set_layer_tensors(cache, idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
    layers = getattr(cache, "layers", None)
    if layers is not None:            # transformers >= 4.56
        layers[idx].keys, layers[idx].values = k, v
    else:
        cache.key_cache[idx], cache.value_cache[idx] = k, v


def _round(x: torch.Tensor, bits: int, scale: str, group: int, clip: float,
           exact_rows: Sequence[int], side: _Side) -> torch.Tensor:
    """Round ``x`` [B, H, T, D]; rows in ``exact_rows`` come back exact.

    For per-channel groups the exact rows are first replaced by a neighbour, so they take
    no part in any range; per-token groups never mix rows, so nothing needs replacing.
    """
    t = x.shape[-2]
    xf = x.float()
    src = xf
    rows = [r for r in exact_rows if 0 <= r < t]
    if rows and scale == "per_channel":
        keep = [r for r in range(t) if r not in rows]
        if not keep:
            return x
        src = xf.clone()
        for r in rows:
            near = min(keep, key=lambda j: (abs(j - r), j))
            src[..., r, :] = xf[..., near, :]
    q = quantize_kv(src, bits, group, symmetric=False, clip_ratio=clip, axis=_AXIS[scale])
    for r in rows:
        q[..., r, :] = xf[..., r, :]
    mask = torch.ones(t, dtype=torch.bool, device=x.device)
    if rows:
        mask[rows] = False
    side.err += float(((q - xf)[..., mask, :] ** 2).sum())
    side.norm += float((xf[..., mask, :] ** 2).sum())
    side.quantized += t - len(rows)
    return q.to(x.dtype)


def make_committed_cache_class(base, cfg: CommittedCacheConfig, stats: CommittedCacheStats):
    """A subclass of ``base`` (the model's ``DynamicCache``) that rounds on write.

    ``base`` is passed in so the subclass follows whatever ``DynamicCache`` the model's
    module imported; a subclass of a different one would not be accepted by it.
    """
    exact = tuple(sorted(set(cfg.exact_positions)))

    class CommittedQuantCache(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._n_rounded: Dict[int, int] = {}   # per layer: leading K tokens rounded

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            start = self.get_seq_length(layer_idx)
            n = int(key_states.shape[-2])
            h, d = int(key_states.shape[1]), int(key_states.shape[-1])
            row_pairs = h * -(-d // min(cfg.group, d))   # per-token pairs for one token
            if layer_idx == 0:
                stats.write_lengths.append(n)
                stats.numbers_per_token = h * d
            stats.k.tokens += n
            stats.v.tokens += n
            local = [p - start for p in exact if start <= p < start + n]

            v = value_states
            if cfg.value_bits < 16:
                v = _round(value_states, cfg.value_bits, "per_token", cfg.group,
                           cfg.clip_ratio, local, stats.v)
                stats.v.pairs += (n - len(local)) * row_pairs

            if cfg.key_bits >= 16:
                return super().update(key_states, v, layer_idx, cache_kwargs)

            if cfg.key_scale == "per_token":
                k = _round(key_states, cfg.key_bits, "per_token", cfg.group,
                           cfg.clip_ratio, local, stats.k)
                stats.k.pairs += (n - len(local)) * row_pairs
                return super().update(k, v, layer_idx, cache_kwargs)

            if cfg.residual == "per_write":
                k = _round(key_states, cfg.key_bits, "per_channel", cfg.group,
                           cfg.clip_ratio, local, stats.k)
                stats.k.pairs += -(-n // cfg.group) * h * d
                return super().update(k, v, layer_idx, cache_kwargs)

            # buffer: append exact, then round every full group of the exact tail.
            k_all, v_all = super().update(key_states, v, layer_idx, cache_kwargs)
            done = self._n_rounded.get(layer_idx, 0)
            full = (k_all.shape[-2] - done) // cfg.group * cfg.group
            if full:
                chunk = k_all[..., done:done + full, :]
                rows = [p - done for p in exact if done <= p < done + full]
                rounded = _round(chunk, cfg.key_bits, "per_channel", cfg.group,
                                 cfg.clip_ratio, rows, stats.k)
                stats.k.pairs += full // cfg.group * h * d
                k_all = torch.cat([k_all[..., :done, :], rounded,
                                   k_all[..., done + full:, :]], dim=-2)
                _set_layer_tensors(self, layer_idx, k_all, v_all)
                self._n_rounded[layer_idx] = done + full
            return k_all, v_all

        def crop(self, max_length: int):
            stats.crops += 1
            for idx, done in list(self._n_rounded.items()):
                self._n_rounded[idx] = min(done, max_length)
            return super().crop(max_length)

    CommittedQuantCache.__name__ = "CommittedQuantCache"
    return CommittedQuantCache


def install_committed_cache(model, cfg: CommittedCacheConfig
                            ) -> Tuple[Callable[[], None], CommittedCacheStats]:
    """Replace the ``DynamicCache`` the model's module builds. Returns (remove, stats).

    For a model whose generation loop constructs the cache itself. Where the caller builds
    it (SDAR's script passes ``past_key_values`` in), use ``make_committed_cache_class``
    on ``transformers.DynamicCache`` and instantiate it directly.
    """
    mod = sys.modules[type(model).__module__]
    base = getattr(mod, "DynamicCache", None)
    if base is None:
        raise RuntimeError(f"{mod.__name__} has no module-level DynamicCache to replace")
    stats = CommittedCacheStats()
    mod.DynamicCache = make_committed_cache_class(base, cfg, stats)

    def remove() -> None:
        mod.DynamicCache = base

    return remove, stats


def check_writes(stats: CommittedCacheStats, block_size: int,
                 expected_tokens: Optional[int] = None) -> List[str]:
    """Self-checks 4 and 5 of the plan: problems found, empty when the run is clean.

    The first write is the prompt (however the model aligns it); every later write must be
    one finished block, the last possibly shorter. A model that writes its current block at
    every step and crops it shows up here as crops or as writes of the wrong size.
    """
    problems = []
    if stats.crops:
        problems.append(f"{stats.crops} crops: the store was written and cut back, so "
                        "the current block may have been rounded")
    body = stats.write_lengths[1:]
    if body:
        wrong = [n for n in body[:-1] if n != block_size]
        if wrong:
            problems.append(f"{len(wrong)} writes not of block size {block_size}: "
                            f"{sorted(set(wrong))[:5]}")
        if body[-1] > block_size:
            problems.append(f"last write {body[-1]} exceeds block size {block_size}")
    if stats.k.tokens != stats.v.tokens:
        problems.append("K and V token counts differ")
    layers = stats.k.tokens // max(stats.stored_tokens, 1)
    if stats.stored_tokens and stats.k.tokens != layers * stats.stored_tokens:
        problems.append("layers wrote different numbers of tokens")
    if expected_tokens is not None and stats.stored_tokens != expected_tokens:
        problems.append(f"store holds {stats.stored_tokens} tokens, text has "
                        f"{expected_tokens}")
    return problems


__all__ = ["CommittedCacheConfig", "CommittedCacheStats", "make_committed_cache_class",
           "install_committed_cache", "check_writes", "KEY_SCALES", "RESIDUALS"]
