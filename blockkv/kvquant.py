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

`impl="kivi"` rounds with the official KIVI code instead of `fake_quant`: the
functions `quant_and_pack_kcache` / `quant_and_pack_vcache` and their unpacking
counterparts from `quant/new_pack.py` of github.com/jy-yuan/KIVI (path in the
KIVI_PATH environment variable, default ~/third_party/KIVI). They pack real 2/4/8-bit
codes and unpack them; the arithmetic is theirs (in the model's dtype, no guard
for a constant group). Entries that come out non-finite (a constant group divides
by zero there) keep their original value and are counted in `kv_nonfinite`.
"""
import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

_KIVI = None


def kivi_ops():
    """The official KIVI module `quant/new_pack.py`, loaded once from KIVI_PATH."""
    global _KIVI
    if _KIVI is None:
        root = Path(os.environ.get("KIVI_PATH", "~/third_party/KIVI")).expanduser()
        src = root / "quant" / "new_pack.py"
        if not src.exists():
            raise FileNotFoundError(f"official KIVI code not found: {src} (set KIVI_PATH)")
        spec = importlib.util.spec_from_file_location("kivi_new_pack", src)
        _KIVI = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_KIVI)
    return _KIVI


def kivi_keys(k, bits, group):
    """Official KIVI: keys per channel over `group` tokens. k: (b, h, n * group, d)."""
    ops = kivi_ops()
    code, scale, mn = ops.quant_and_pack_kcache(k.contiguous(), group, bits)
    return ops.unpack_and_dequant_kcache(code, scale, mn, group, bits).to(k.dtype)


def kivi_values(v, bits, group):
    """Official KIVI: values per token over `group` channels. v: (b, h, t, d)."""
    ops = kivi_ops()
    code, scale, mn = ops.quant_and_pack_vcache(v.contiguous(), group, bits)
    return ops.unpack_and_dequant_vcache(code, scale, mn, group, bits).to(v.dtype)


def cache_layers(cache):
    """[(keys, values)] per layer, each (batch, kv_heads, seq, head_dim)."""
    if hasattr(cache, "key_cache"):          # transformers <= 4.55 DynamicCache
        return list(zip(cache.key_cache, cache.value_cache))
    return [(layer.keys, layer.values) for layer in cache.layers]


def fake_quant(x, bits, dim, shared=False):
    """Quantize-dequantize `x` with one (min, step) per slice along `dim`.

    Computed in float32; min and step are rounded to x's dtype first, as they
    would be stored next to the codes. `shared`: one (min, step) for the whole
    batch instead of one per sample (official KIVI keeps them per sample).
    """
    xf = x.float()
    levels = 2 ** bits - 1
    dims = (0, dim) if shared else dim
    mn = xf.amin(dims, keepdim=True)
    step = (xf.amax(dims, keepdim=True) - mn) / levels
    mn = mn.to(x.dtype).float()
    step = step.to(x.dtype).float()
    step = torch.where(step > 0, step, torch.ones_like(step))   # constant slice: exact
    q = ((xf - mn) / step).round().clamp(0, levels)
    return (q * step + mn).to(x.dtype)


def quant_keys(k, bits, group, mode, shared=False):
    """k: (b, h, n * group, d) -> same shape, quantized."""
    b, h, t, d = k.shape
    if mode == "channel":
        return fake_quant(k.reshape(b, h, t // group, group, d), bits, 3, shared).reshape(b, h, t, d)
    return quant_tokens(k, bits, group, shared)


def quant_tokens(v, bits, group, shared=False):
    """v: (b, h, t, d), groups of `group` channels within each token."""
    b, h, t, d = v.shape
    assert d % group == 0, f"head_dim {d} is not a multiple of the group {group}"
    return fake_quant(v.reshape(b, h, t, d // group, group), bits, 4, shared).reshape(b, h, t, d)


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
    shared: bool = False                   # one (min, step) per batch instead of per sample
    impl: str = "ours"                     # "ours": fake_quant; "kivi": the official KIVI code
    done: int = None                       # first position not yet quantized
    nonfinite: int = 0                     # impl="kivi": entries that came out non-finite and were kept
    _err: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.key_mode in ("channel", "token")
        assert self.only in (None, "k", "v")
        assert self.impl in ("ours", "kivi")
        if self.impl == "kivi":
            assert self.key_mode == "channel" and not self.shared, "official KIVI: keys per channel, per sample"
            assert self.bits in (2, 4, 8, 16), "official KIVI packs 2, 4 or 8 bits"
        self.done = self.sink

    def _keys(self, ks):
        if self.impl == "kivi":
            return self._kept(ks, kivi_keys(ks, self.bits, self.group))
        return quant_keys(ks, self.bits, self.group, self.key_mode, self.shared)

    def _values(self, vs):
        if self.impl == "kivi":
            return self._kept(vs, kivi_values(vs, self.bits, self.group))
        return quant_tokens(vs, self.bits, self.group, self.shared)

    def _kept(self, x, xq):
        bad = ~torch.isfinite(xq)
        n = int(bad.sum())
        if n:
            self.nonfinite += n
            xq = torch.where(bad, x, xq)
        return xq

    def __call__(self, cache, start, end):
        n = (end - self.window - self.done) // self.group
        if n <= 0:
            return
        lo, hi = self.done, self.done + n * self.group
        for k, v in cache_layers(cache):
            ks, vs = k[:, :, lo:hi], v[:, :, lo:hi]
            kq = self._keys(ks) if self._does("k") else ks
            vq = self._values(vs) if self._does("v") else vs
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
        if self.impl == "kivi":
            out["kv_nonfinite"] = self.nonfinite
        return out
