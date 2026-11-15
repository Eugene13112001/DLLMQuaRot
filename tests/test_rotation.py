import torch

from blockkv.kvquant import KVQuant, hadamard, quant_keys, quant_tokens, rotated


def test_hadamard_is_orthogonal():
    r = hadamard(128, "cpu")
    assert torch.allclose(r @ r.T, torch.eye(128), atol=1e-5)


def test_rotation_without_rounding_is_identity():
    x = torch.randn(1, 2, 32, 128)
    assert torch.allclose(rotated(lambda t: t, x), x, atol=1e-5)


def test_rotation_beats_plain_per_token_on_outlier_channels():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 64, 128)
    x[..., 7] *= 30                                   # a few outlier channels, as in keys
    x[..., 50] *= 20
    err = lambda xq: float((xq - x).pow(2).sum() / x.pow(2).sum())
    assert err(rotated(lambda t: quant_tokens(t, 2, 32), x)) < err(quant_tokens(x, 2, 32))
    # one group over the whole head (QuaRot's setting): without rotation the outliers ruin it
    assert err(rotated(lambda t: quant_tokens(t, 2, 128), x)) < err(quant_tokens(x, 2, 128)) / 5


def test_kvquant_rotate_hook_measures_error_in_original_basis():
    class Layer:
        def __init__(self):
            self.keys = torch.randn(1, 2, 64, 128)
            self.values = torch.randn(1, 2, 64, 128)

    class Cache:
        layers = [Layer()]
    c = Cache()
    k0 = c.layers[0].keys.clone()
    q = KVQuant(2, 32, key_mode="token", rotate=True)
    q(c, 0, 64)
    s = q.summary(64)
    assert 0 < s["kv_relerr_k"] < 0.5 and not torch.equal(c.layers[0].keys, k0)
    assert q.honest_bits == 3.0
