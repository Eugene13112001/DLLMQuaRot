"""Compatibility shim for LLaDA's remote code under a modern transformers.

The shim asserts something about the checkpoint ("nothing is tied"), so the
guard that refuses to assert it wrongly is the part worth testing.
"""

from __future__ import annotations

import types

import pytest

from dllmquant.models.base import _dtype_kwargs, ensure_tied_weights_attr


class _Cfg:
    model_path = "fake/model"


def _fake_transformers(monkeypatch, *, has_attr: bool, tied):
    """Stand in for the transformers module without importing a real model."""
    import transformers

    class FakeBase:
        pass

    if has_attr:
        FakeBase.all_tied_weights_keys = {}

    fake_conf = types.SimpleNamespace(weight_tying=tied)

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(*a, **k):
            return fake_conf

    monkeypatch.setattr(transformers, "PreTrainedModel", FakeBase)
    monkeypatch.setattr(transformers, "AutoConfig", FakeAutoConfig)
    return FakeBase


def test_shim_installs_when_the_model_ties_nothing(monkeypatch):
    base = _fake_transformers(monkeypatch, has_attr=False, tied=False)
    assert ensure_tied_weights_attr(_Cfg()) is True
    assert base.all_tied_weights_keys == {}


def test_shim_refuses_when_weights_are_tied(monkeypatch):
    """Claiming nothing is tied would leave the output head uninitialised."""
    _fake_transformers(monkeypatch, has_attr=False, tied=True)
    with pytest.raises(RuntimeError) as exc:
        ensure_tied_weights_attr(_Cfg())
    assert "transformers==4.38.2" in str(exc.value)


def test_shim_is_a_no_op_when_transformers_already_provides_it(monkeypatch):
    _fake_transformers(monkeypatch, has_attr=True, tied=False)
    assert ensure_tied_weights_attr(_Cfg()) is False


def test_shim_checks_the_llama_style_config_key_too(monkeypatch):
    """OLMo-style configs say `weight_tying`; Llama-style say
    `tie_word_embeddings`. Both must be honoured."""
    import transformers

    class FakeBase:
        pass

    conf = types.SimpleNamespace(tie_word_embeddings=True)

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(*a, **k):
            return conf

    monkeypatch.setattr(transformers, "PreTrainedModel", FakeBase)
    monkeypatch.setattr(transformers, "AutoConfig", FakeAutoConfig)

    with pytest.raises(RuntimeError):
        ensure_tied_weights_attr(_Cfg())


def test_version_guard_rejects_a_release_past_the_window(monkeypatch):
    """Fail in a second, not after downloading 16 GB and crashing inside
    modeling_utils with a traceback that names none of the real cause."""
    import transformers

    from dllmquant.models.base import check_transformers_version
    from dllmquant.models.llada import LLaDAAdapter

    monkeypatch.setattr(transformers, "__version__", "5.14.1")
    with pytest.raises(RuntimeError) as exc:
        check_transformers_version(
            LLaDAAdapter.TRANSFORMERS_MIN, LLaDAAdapter.TRANSFORMERS_MAX
        )
    msg = str(exc.value)
    assert "4.46.3" in msg
    assert "--allow-untested-transformers" in msg


def test_version_guard_rejects_a_release_below_the_window(monkeypatch):
    """The two LLaDA families pull in opposite directions: 4.46 is fine for
    1.5 and too old for 2.0, whose remote code needs `dynamic_rope_update`."""
    import transformers

    from dllmquant.models.base import check_transformers_version
    from dllmquant.models.llada2_moe import LLaDA2MoEAdapter

    monkeypatch.setattr(transformers, "__version__", "4.46.3")
    with pytest.raises(RuntimeError) as exc:
        check_transformers_version(
            LLaDA2MoEAdapter.TRANSFORMERS_MIN, LLaDA2MoEAdapter.TRANSFORMERS_MAX
        )
    msg = str(exc.value)
    assert "older" in msg
    assert "4.57.1" in msg


def test_the_two_llada_windows_do_not_overlap(monkeypatch):
    """One venv cannot serve both models. Asserting it here keeps the fact
    from being rediscovered as a confusing ImportError."""
    import transformers

    from dllmquant.models.base import check_transformers_version
    from dllmquant.models.llada import LLaDAAdapter
    from dllmquant.models.llada2_moe import LLaDA2MoEAdapter

    for v in ("4.38.2", "4.46.3", "4.56.2", "4.57.1", "5.0.0"):
        monkeypatch.setattr(transformers, "__version__", v)
        ok = []
        for adapter in (LLaDAAdapter, LLaDA2MoEAdapter):
            try:
                check_transformers_version(
                    adapter.TRANSFORMERS_MIN, adapter.TRANSFORMERS_MAX
                )
                ok.append(adapter.__name__)
            except RuntimeError:
                pass
        assert len(ok) == 1, f"{v} accepted by {ok}"


def test_version_guard_passes_releases_inside_the_window(monkeypatch):
    import transformers

    from dllmquant.models.base import check_transformers_version
    from dllmquant.models.llada import LLaDAAdapter

    for v in ("4.38.2", "4.46.3"):
        monkeypatch.setattr(transformers, "__version__", v)
        got = check_transformers_version(
            LLaDAAdapter.TRANSFORMERS_MIN, LLaDAAdapter.TRANSFORMERS_MAX
        )
        assert got == v


def test_version_guard_can_be_waived(monkeypatch):
    import transformers

    from dllmquant.models.base import check_transformers_version
    from dllmquant.models.llada import LLaDAAdapter

    monkeypatch.setattr(transformers, "__version__", "5.14.1")
    got = check_transformers_version(
        LLaDAAdapter.TRANSFORMERS_MIN,
        LLaDAAdapter.TRANSFORMERS_MAX,
        strict=False,
    )
    assert got == "5.14.1"


def test_dtype_kwarg_follows_the_transformers_rename(monkeypatch):
    """`torch_dtype` became `dtype` in 4.56."""
    import torch
    import transformers

    monkeypatch.setattr(transformers, "__version__", "4.46.2")
    assert _dtype_kwargs(torch.bfloat16) == {"torch_dtype": torch.bfloat16}

    monkeypatch.setattr(transformers, "__version__", "4.57.0.dev0")
    assert _dtype_kwargs(torch.bfloat16) == {"dtype": torch.bfloat16}


def test_dtype_kwarg_survives_an_unparseable_version(monkeypatch):
    import torch
    import transformers

    monkeypatch.setattr(transformers, "__version__", "weird-build")
    assert "torch_dtype" in _dtype_kwargs(torch.bfloat16)
