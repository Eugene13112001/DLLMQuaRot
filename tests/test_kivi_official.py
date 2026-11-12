"""impl="kivi": the official KIVI code inside our hook.

The wiring tests run anywhere with a stub `quant/new_pack.py`. The comparison with
the real KIVI code runs only where it is installed (KIVI_PATH or ~/third_party/KIVI):
    KIVI_PATH=~/third_party/KIVI python -m pytest tests/test_kivi_official.py -v
"""
import os
from pathlib import Path

import pytest
import torch

from blockkv import kvquant
from blockkv.kvquant import KVQuant, quant_keys, quant_tokens

STUB = '''
import torch

def quant_and_pack_kcache(k, group_size, bits):
    return k.clone(), None, None

def unpack_and_dequant_kcache(code, scale, mn, group_size, bits):
    out = code.half()
    out[0, 0, 0, 0] = float("nan")       # what a constant group gives in the official code
    return out

def quant_and_pack_vcache(v, group_size, bits):
    return v.clone(), None, None

def unpack_and_dequant_vcache(code, scale, mn, group_size, bits):
    return code.half() + 1
'''


class Cache:
    def __init__(self, k, v):
        self.key_cache, self.value_cache = [k], [v]


@pytest.fixture
def stub(tmp_path, monkeypatch):
    (tmp_path / "quant").mkdir()
    (tmp_path / "quant" / "new_pack.py").write_text(STUB)
    monkeypatch.setenv("KIVI_PATH", str(tmp_path))
    monkeypatch.setattr(kvquant, "_KIVI", None)
    yield
    kvquant._KIVI = None


def test_kivi_impl_calls_the_official_functions_and_keeps_non_finite(stub):
    k, v = torch.randn(1, 2, 64, 64), torch.randn(1, 2, 64, 64)
    k0, v0 = k.clone(), v.clone()
    q = KVQuant(2, impl="kivi")
    q(Cache(k, v), 0, 64)
    assert q.done == 64 and q.nonfinite == 1
    assert k.dtype == torch.float32                      # written back in the cache's dtype
    assert k[0, 0, 0, 0] == k0[0, 0, 0, 0]               # the non-finite entry kept its value
    assert torch.allclose(k[0, 1], k0[0, 1].half().float())
    assert torch.allclose(v, (v0.half() + 1).float())
    assert q.summary(64)["kv_nonfinite"] == 1


def test_kivi_impl_refuses_what_kivi_has_not():
    with pytest.raises(AssertionError):
        KVQuant(3, impl="kivi")
    with pytest.raises(AssertionError):
        KVQuant(2, key_mode="token", impl="kivi")


def test_missing_kivi_code_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("KIVI_PATH", str(tmp_path))
    monkeypatch.setattr(kvquant, "_KIVI", None)
    with pytest.raises(FileNotFoundError, match="KIVI_PATH"):
        KVQuant(2, impl="kivi")(Cache(torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32)), 0, 32)


# ---------- against the real KIVI code (server only) ----------

ROOT = Path(os.environ.get("KIVI_PATH", "~/third_party/KIVI")).expanduser()
real = pytest.mark.skipif(not (ROOT / "quant" / "new_pack.py").exists() or not torch.cuda.is_available(),
                          reason="official KIVI code or GPU not available")


def levels_apart(a, b, x, bits, step_of):
    """Share of entries where two dequantized tensors differ by about one level or more."""
    return float(((a.float() - b.float()).abs() > 0.5 * step_of(x)).float().mean())


@real
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bits", [2, 4])
def test_ours_matches_official_kivi(dtype, bits):
    kvquant._KIVI = None
    os.environ["KIVI_PATH"] = str(ROOT)
    torch.manual_seed(0)
    k = (torch.randn(1, 8, 256, 128, device="cuda") * 3 + torch.randn(1, 8, 1, 128, device="cuda") * 5).to(dtype)
    v = torch.randn(1, 8, 256, 128, device="cuda").to(dtype)

    def key_step(x):     # per channel over 32 tokens
        g = x.float().reshape(1, 8, 8, 32, 128)
        return ((g.amax(3, keepdim=True) - g.amin(3, keepdim=True)) / (2 ** bits - 1)).expand_as(g).reshape(x.shape)

    def value_step(x):   # per token over 32 channels
        g = x.float().reshape(1, 8, 256, 4, 32)
        return ((g.amax(4, keepdim=True) - g.amin(4, keepdim=True)) / (2 ** bits - 1)).expand_as(g).reshape(x.shape)

    ko, vo = kvquant.kivi_keys(k, bits, 32), kvquant.kivi_values(v, bits, 32)
    kq, vq = quant_keys(k, bits, 32, "channel"), quant_tokens(v, bits, 32)
    assert torch.isfinite(ko).all() and torch.isfinite(vo).all()
    # Same formula; only rounding of the arithmetic differs (their model dtype, our float32),
    # so a value on the edge between two levels may land on the other one.
    dk, dv = levels_apart(ko, kq, k, bits, key_step), levels_apart(vo, vq, v, bits, value_step)
    print(f"{dtype} {bits} bit: keys {100 * dk:.3f}% / values {100 * dv:.3f}% entries one level apart")
    assert dk < 0.01 and dv < 0.01
    # Same error against the original, to well under a percent.
    for x, o, q in ((k, ko, kq), (v, vo, vq)):
        eo = (o.float() - x.float()).pow(2).mean()
        eq = (q.float() - x.float()).pow(2).mean()
        assert abs(float(eo / eq) - 1) < 0.02
