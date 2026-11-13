"""impl="noise": uniform noise of the rounding step instead of rounding (control)."""
import torch

from blockkv.kvquant import KVQuant, quant_keys, quant_tokens


class Cache:
    def __init__(self, k, v):
        self.key_cache, self.value_cache = [k], [v]


def data(seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, 8, 512, 128, generator=g) * 3 + torch.randn(1, 8, 1, 128, generator=g) * 5
    v = torch.randn(1, 8, 512, 128, generator=g)
    return k, v


def test_noise_error_has_the_size_of_the_rounding_error():
    k, v = data()
    g = torch.Generator().manual_seed(0)
    for bits in (2, 3, 4):
        for x, f in ((k, lambda x, n: quant_keys(x, bits, 32, "channel", noise=n)),
                     (v, lambda x, n: quant_tokens(x, bits, 32, noise=n))):
            e_round = (f(x, None) - x).pow(2).mean()
            e_noise = (f(x, g) - x).pow(2).mean()
            assert abs(float(e_noise / e_round) - 1) < 0.1, (bits, float(e_noise / e_round))


def test_noise_is_unbiased_and_not_on_levels():
    k, _ = data()
    g = torch.Generator().manual_seed(0)
    kn = quant_keys(k, 2, 32, "channel", noise=g)
    kq = quant_keys(k, 2, 32, "channel")
    assert abs(float((kn - k).mean())) < 1e-2                 # random, no systematic shift
    levels = lambda x: len(torch.unique(x[0, 0, :32, 0]))
    assert levels(kq) <= 4 and levels(kn) > 4                  # rounding gives 4 values per group, noise does not


def test_noise_hook_is_deterministic_and_counts_error():
    outs = []
    for _ in range(2):
        k, v = data()
        q = KVQuant(2, impl="noise", seed=3)
        q(Cache(k, v), 0, 512)
        outs.append((k, v, q.summary(512)))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
    s = outs[0][2]
    assert s["kv_quantized"] == 512 and s["kv_relerr_k"] > 0 and s["kv_relerr_v"] > 0
