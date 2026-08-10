"""Trajectory divergence: letting the quantized model walk on its own.

The metric this replaces reads agreement on states drawn from the FP16 model's
path.  On the real model that read 88.9% for a configuration solving 26% of
GSM8K and 91.2% for one solving 78% -- so a 2-point spread stood in for a
52-point one.  These tests check that this metric does register what that one
missed.
"""

from __future__ import annotations

import torch

from dllmquant.eval import capture_reference, measure_divergence

from test_pipeline_e2e import TinyAdapter, _config, _prompts


def _adapter(cfg=None):
    torch.manual_seed(0)
    a = TinyAdapter(cfg or _config())
    a.load()
    return a


def test_unchanged_model_never_diverges_from_itself():
    cfg = _config()
    adapter = _adapter(cfg)
    prompts = _prompts(adapter, 3)

    refs = capture_reference(adapter, prompts, cfg.tmas)
    result = measure_divergence(adapter, refs, cfg.tmas)

    assert result.n_prompts == 3
    assert result.exact_match == 1.0
    assert result.final_agreement == 1.0
    assert result.never_diverged == 3


def _damage(adapter, scale: float = 0.5) -> None:
    """Perturb the weights outright.

    The point under test is whether the metric registers a changed model, not
    whether some particular bit width happens to damage a 32-channel toy with
    a 64-token vocabulary -- that toy shrugs off 2-bit RTN, which says nothing
    about the metric.
    """
    torch.manual_seed(7)
    with torch.no_grad():
        for p in adapter.model.parameters():
            p.add_(scale * torch.randn_like(p))


def test_a_damaged_model_diverges():
    cfg = _config()
    adapter = _adapter(cfg)
    prompts = _prompts(adapter, 3)
    refs = capture_reference(adapter, prompts, cfg.tmas)

    _damage(adapter)

    result = measure_divergence(adapter, refs, cfg.tmas)
    assert result.final_agreement < 1.0
    assert result.never_diverged < 3


def test_divergence_step_is_recorded_with_its_mask_ratio():
    """Where a trajectory splits matters: early splits are unrecoverable,
    because a committed token is never revised."""
    cfg = _config()
    adapter = _adapter(cfg)
    prompts = _prompts(adapter, 2)
    refs = capture_reference(adapter, prompts, cfg.tmas)

    _damage(adapter)

    result = measure_divergence(adapter, refs, cfg.tmas)
    for entry in result.per_prompt:
        if entry["first_divergence_step"] >= 0:
            assert 0 <= entry["first_divergence_step"] < cfg.tmas.steps
            r = entry["divergence_mask_ratio"]
            assert r != r or 0.0 <= r <= 1.0  # NaN or a valid ratio


def test_reference_records_every_step():
    cfg = _config()
    adapter = _adapter(cfg)
    refs = capture_reference(adapter, _prompts(adapter, 1), cfg.tmas)

    ref = refs[0]
    assert len(ref.states) == cfg.tmas.steps
    assert len(ref.mask_ratios) == cfg.tmas.steps
    assert ref.mask_ratios[0] == 1.0
    assert ref.final_ids.numel() == ref.prompt_len + cfg.tmas.gen_length


def test_summary_is_printable():
    cfg = _config()
    adapter = _adapter(cfg)
    refs = capture_reference(adapter, _prompts(adapter, 2), cfg.tmas)
    text = measure_divergence(adapter, refs, cfg.tmas).summary()
    assert "trajectory divergence" in text
    assert "final token agreement" in text


def test_empty_reference_is_refused():
    import pytest

    adapter = _adapter()
    with pytest.raises(ValueError):
        measure_divergence(adapter, [], _config().tmas)
