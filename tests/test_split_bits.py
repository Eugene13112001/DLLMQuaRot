import torch

from blockkv.kvquant import KVQuant


class Layer:
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.keys = torch.randn(1, 2, 64, 128, generator=g)
        self.values = torch.randn(1, 2, 64, 128, generator=g)


class Cache:
    def __init__(self):
        self.layers = [Layer()]


def run(**kw):
    c = Cache()
    k0, v0 = c.layers[0].keys.clone(), c.layers[0].values.clone()
    q = KVQuant(kw.pop("bits", 2), 32, **kw)
    q(c, 0, 64)
    return q, c.layers[0].keys, c.layers[0].values, k0, v0


def test_default_is_the_same_bits_for_both():
    q, *_ = run()
    assert (q.k_bits, q.v_bits, q.honest_bits) == (2, 2, 3.0)


def test_k4_v2_spends_more_on_keys():
    q, k, v, k0, v0 = run(bits=4, k_bits=4, v_bits=2)
    s = q.summary(64)
    assert q.honest_bits == (5 + 3) / 2
    assert s["kv_relerr_k"] < s["kv_relerr_v"]           # 4-bit keys round finer than 2-bit values


def test_16_leaves_that_part_untouched():
    q, k, v, k0, v0 = run(bits=2, k_bits=16, v_bits=2)
    assert torch.equal(k, k0) and not torch.equal(v, v0)
    assert q.honest_bits == (16 + 3) / 2
