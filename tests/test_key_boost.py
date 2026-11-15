import torch

from blockkv.kvquant import KVQuant, boosted_keys, quant_keys


def test_boost_lowers_key_error_on_outlier_channels():
    g = torch.Generator().manual_seed(0)
    k = torch.randn(1, 2, 64, 128, generator=g)
    k[..., 5] *= 25
    k[..., 77] *= 15
    err = lambda kq: float((kq - k).pow(2).sum() / k.pow(2).sum())
    plain = err(quant_keys(k, 2, 32, "channel"))
    boost = err(boosted_keys(k, 2, 32, 0.0625, 8))           # 8 of 128 channels at 8 bits
    assert boost < plain / 3


def test_zero_like_boost_at_same_bits_changes_nothing():
    k = torch.randn(1, 1, 32, 128)
    assert torch.equal(boosted_keys(k, 2, 32, 0.0625, 2), quant_keys(k, 2, 32, "channel"))


def test_honest_bits_count_wider_codes_and_indices():
    q = KVQuant(2, 32, key_boost=0.0625, key_boost_bits=8)
    nb = 8
    k_bits = 2 + 1 + nb / 128 * 6 + nb * 7 / (32 * 128)
    assert abs(q.honest_bits - (k_bits + 3) / 2) < 1e-9
