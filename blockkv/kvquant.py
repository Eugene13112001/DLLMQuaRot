"""KV-cache quantization at block commit (stage 1.2).

KIVI-style fake quantization: dynamic, asymmetric min-max, round to nearest.
The cache is quantized and dequantized in place, so the model reads back exactly
what a real 2/3/4-bit cache would return; the memory itself is not saved.

Variant A (KIVI, MaskAhead): finished blocks go into the cache in 16 bit and wait
there until a whole group of `group` tokens has gathered; then the group is
quantized, once, and never touched again. The group is the same for every block
size, so blocks are compared at the same granularity and the same honest bits:
`bits + 32 / group` per number (min and step, 16 bit each, per group).

  keys    per channel: one (min, step) per channel over `group` tokens
          (`key_mode="token"`: like the values instead);
  values  per token:   one (min, step) per token over `group` channels.

`window` newest tokens always stay in 16 bit (KIVI's residual); `sink` first
tokens are never quantized. `only="k"` or `"v"` quantizes the keys or the values
alone. `bits=16` runs the hook without rounding anything: the cache must come
out bit-identical, which checks the hook itself. The current block is never in the cache, so it is
never quantized either.
"""
from dataclasses import dataclass, field

import torch


def cache_layers(cache):
    """[(keys, values)] per layer, each (batch, kv_heads, seq, head_dim)."""
    if hasattr(cache, "key_cache"):          # transformers <= 4.55 DynamicCache
        return list(zip(cache.key_cache, cache.value_cache))
    return [(layer.keys, layer.values) for layer in cache.layers]


def fake_quant(x, bits, dim):
    """Quantize-dequantize `x` with one (min, step) per slice along `dim`.

    Computed in float32; min and step are rounded to x's dtype first, as they
    would be stored next to the codes.
    """
    xf = x.float()
    levels = 2 ** bits - 1
    mn = xf.amin(dim, keepdim=True)
    step = (xf.amax(dim, keepdim=True) - mn) / levels
    mn = mn.to(x.dtype).float()
    step = step.to(x.dtype).float()
    step = torch.where(step > 0, step, torch.ones_like(step))   # constant slice: exact
    q = ((xf - mn) / step).round().clamp(0, levels)
    return (q * step + mn).to(x.dtype)


def quant_keys(k, bits, group, mode):
    """k: (b, h, n * group, d) -> same shape, quantized."""
    b, h, t, d = k.shape
    if mode == "channel":
        return fake_quant(k.reshape(b, h, t // group, group, d), bits, 3).reshape(b, h, t, d)
    return quant_tokens(k, bits, group)


def quant_tokens(v, bits, group):
    """v: (b, h, t, d), groups of `group` channels within each token."""
    b, h, t, d = v.shape
    assert d % group == 0, f"head_dim {d} is not a multiple of the group {group}"
    return fake_quant(v.reshape(b, h, t, d // group, group), bits, 4).reshape(b, h, t, d)


@dataclass
class KVQuant:
    """`on_store` hook. Make a fresh one for every sample: it remembers how far
    the cache is quantized."""
    bits: int
    group: int = 32
    window: int = 0
    sink: int = 0
    key_mode: str = "channel"
    only: str = None                       # None: keys and values; "k" or "v": that one alone
    done: int = None                       # first position not yet quantized
    _err: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.key_mode in ("channel", "token")
        assert self.only in (None, "k", "v")
        self.done = self.sink

    def __call__(self, cache, start, end):
        n = (end - self.window - self.done) // self.group
        if n <= 0:
            return
        lo, hi = self.done, self.done + n * self.group
        for k, v in cache_layers(cache):
            ks, vs = k[:, :, lo:hi], v[:, :, lo:hi]
            kq = quant_keys(ks, self.bits, self.group, self.key_mode) if self._does("k") else ks
            vq = quant_tokens(vs, self.bits, self.group) if self._does("v") else vs
            self._acc("k", ks, kq)
            self._acc("v", vs, vq)
            k[:, :, lo:hi] = kq
            v[:, :, lo:hi] = vq
        self.done = hi

    def _does(self, name):
        return self.bits < 16 and self.only in (None, name)

    def _acc(self, name, x, xq):
        x = x.float()
        e = self._err.setdefault(name, torch.zeros(2, device=x.device))
        e[0] += (xq.float() - x).pow(2).sum()
        e[1] += x.pow(2).sum()

    @property
    def honest_bits(self):
        """Bits per cached number, keys and values averaged; 16 for what stays 16 bit."""
        per = [self.bits + 32 / self.group if self._does(n) else 16 for n in ("k", "v")]
        return sum(per) / 2

    def summary(self, cache_len):
        """Per-sample numbers for the output row."""
        q = self.done - self.sink
        out = dict(kv_quantized=q, kv_tokens=cache_len,
                   kv_bits_avg=round((q * self.honest_bits + (cache_len - q) * 16) / max(cache_len, 1), 3))
        for name, e in self._err.items():
            out[f"kv_relerr_{name}"] = round(float(e[0] / e[1].clamp_min(1e-30)), 6)
        return out
