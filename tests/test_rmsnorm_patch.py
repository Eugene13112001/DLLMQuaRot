import sys
import types

import torch

from blockkv.run_gsm8k import patch_rmsnorm, torch_rms_norm


def test_torch_rms_norm_matches_the_formula():
    x = torch.randn(3, 5, 64).half()
    w = torch.randn(64).half()
    xf = x.float()
    ref = (xf / torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).half()
    y = torch_rms_norm(x, weight=w, bias=None, eps=1e-6)
    assert y.dtype == torch.float16 and torch.allclose(y.float(), ref.float(), atol=1e-3)


def test_patch_swaps_the_global_the_model_code_calls():
    mod = types.ModuleType("fake_modeling_sdar")
    exec("def flash_rms_norm(*a, **k):\n    raise RuntimeError('triton')\n"
         "class Norm:\n    def __call__(self, x, w):\n        return flash_rms_norm(x, weight=w, bias=None, eps=1e-6)\n"
         "class Model:\n    pass\n", mod.__dict__)
    mod.Model.__module__ = mod.__name__
    sys.modules[mod.__name__] = mod
    try:
        assert patch_rmsnorm(mod.Model())
        x, w = torch.randn(2, 8), torch.ones(8)
        assert torch.allclose(mod.Norm()(x, w), torch_rms_norm(x, w))
    finally:
        del sys.modules[mod.__name__]


def test_patch_reports_models_without_it():
    class Plain:
        pass
    assert not patch_rmsnorm(Plain())
