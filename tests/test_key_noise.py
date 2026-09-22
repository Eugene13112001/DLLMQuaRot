"""The structureless control: error of a chosen size and no shape.

Thesis 4 has LLaDA2.0-mini and Fast-dLLM-v2 carrying the same centered logit error at
four bits per token and scoring 91.5 and 0.0. Either the shape of the quantizer's error
is what separates them, or one model is the more fragile of the two at any error of that
size. Gaussian noise of a matched dose is the second hypothesis made testable, so the
dose has to mean the same thing at every width and axis, and zero has to be exactly the
run we already measured.
"""

from __future__ import annotations

import pytest
import torch

from dllmquant.cache import BlockKVCache, KVCacheConfig, add_key_noise


def keys(seed=0):
    torch.manual_seed(seed)
    k = torch.randn(1, 2, 32, 8)
    k[:, :, :, 3] *= 10.0        # a loud channel, as a gain or a bias makes
    return k


def test_zero_is_the_identity():
    k = keys()
    assert torch.equal(add_key_noise(k, k, 0.0), k)


def test_dose_is_in_units_of_the_true_rms_per_head():
    k = keys()
    k = torch.cat([k, 5.0 * k], dim=1)        # two heads of very different size
    out = add_key_noise(torch.zeros_like(k), k, 0.1)
    for h in range(k.shape[1]):
        rms = k[:, h].pow(2).mean().sqrt()
        assert out[:, h].std() == pytest.approx(0.1 * float(rms), rel=0.15)


def test_the_dose_does_not_depend_on_what_it_is_added_to():
    """Measured against the true keys, so 16 bits and 4 bits carry the same dose."""
    k = keys().double()
    rounded = torch.round(k)
    torch.manual_seed(1)
    a = add_key_noise(k, k, 0.05) - k
    torch.manual_seed(1)
    b = add_key_noise(rounded, k, 0.05) - rounded
    assert torch.allclose(a, b, atol=1e-9)


def test_the_cache_leaves_values_alone_and_keys_noisy():
    cfg = KVCacheConfig(enabled=True, decoded_bits=16, masked_bits=16, key_noise=0.05)
    cache = BlockKVCache(cfg, n_layers=1)
    k, v = keys(), keys(1)
    kq, vq = cache.write(0, k, v)
    assert torch.equal(vq, v)                      # V untouched at 16 bits
    assert not torch.equal(kq, k)
    rel = float((kq - k).norm() / k.norm())
    assert 0.02 < rel < 0.1


def test_a_negative_dose_is_refused():
    with pytest.raises(ValueError, match="key_noise"):
        KVCacheConfig(enabled=True, key_noise=-0.1)
