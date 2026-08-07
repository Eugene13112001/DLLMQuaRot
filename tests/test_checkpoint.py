"""Resuming a quantization run from per-block checkpoints.

The property that matters: resuming must give the SAME model as an
uninterrupted run. A checkpoint that merely lets the script finish, while
producing different weights, is worse than no checkpoint at all.
"""

from __future__ import annotations

import pytest
import torch

from dllmquant.checkpoint import BlockCheckpoints, config_fingerprint
from dllmquant.config import IAAQConfig, QuantConfig, TMASConfig
from dllmquant.modules import find_quant_linears
from dllmquant.pipeline import DLLMQuantPipeline

from test_pipeline_e2e import TinyAdapter, _config, _prompts


def _run(cfg, ckpt_dir="", state=None):
    torch.manual_seed(0)
    adapter = TinyAdapter(cfg)
    if state is not None:
        adapter.model.load_state_dict(state)
    adapter.load()
    cfg.checkpoint_dir = str(ckpt_dir)
    report = DLLMQuantPipeline(cfg, adapter).run(_prompts(adapter), verbose=False)
    weights = {
        n: l.weight.detach().clone()
        for n, l in find_quant_linears(adapter.model).items()
    }
    return adapter, report, weights


def test_resume_reproduces_an_uninterrupted_run(tmp_path):
    torch.manual_seed(0)
    base = TinyAdapter(_config())
    state = {k: v.clone() for k, v in base.model.state_dict().items()}

    # Reference: no checkpoints at all.
    _, _, want = _run(_config(), state=state)

    # First pass writes checkpoints for every block.
    d = tmp_path / "ckpt"
    _, r1, _ = _run(_config(), ckpt_dir=d, state=state)
    assert r1.restored_blocks == set()
    assert (d / "snapshots.pt").exists()
    assert (d / "block_0000.pt").exists()

    # Second pass restores them all and must land on identical weights.
    _, r2, got = _run(_config(), ckpt_dir=d, state=state)
    assert r2.restored_blocks == {0, 1}
    assert set(got) == set(want)
    for name in want:
        assert torch.equal(got[name], want[name]), name


def test_resume_after_losing_the_last_block(tmp_path):
    """The realistic crash: some blocks done, the rest not."""
    torch.manual_seed(0)
    base = TinyAdapter(_config())
    state = {k: v.clone() for k, v in base.model.state_dict().items()}
    _, _, want = _run(_config(), state=state)

    d = tmp_path / "ckpt"
    _run(_config(), ckpt_dir=d, state=state)
    (d / "block_0001.pt").unlink()  # simulate dying midway through block 1

    _, report, got = _run(_config(), ckpt_dir=d, state=state)
    assert report.restored_blocks == {0}
    for name in want:
        assert torch.equal(got[name], want[name]), name


def test_layer_reports_survive_a_resume(tmp_path):
    d = tmp_path / "ckpt"
    _, r1, _ = _run(_config(), ckpt_dir=d)
    _, r2, _ = _run(_config(), ckpt_dir=d)
    assert len(r2.layers) == len(r1.layers)
    assert {l.name for l in r2.layers} == {l.name for l in r1.layers}


def test_value_quantizer_is_restored_not_recomputed(tmp_path):
    d = tmp_path / "ckpt"
    a1, _, _ = _run(_config(), ckpt_dir=d)
    scales1 = {
        n: l.out_quantizer.scale.clone()
        for n, l in find_quant_linears(a1.model).items()
        if l.out_quantizer is not None
    }
    assert scales1, "IA-AQ never attached, so this test proves nothing"

    a2, _, _ = _run(_config(), ckpt_dir=d)
    for n, l in find_quant_linears(a2.model).items():
        if n in scales1:
            assert l.out_quantizer is not None
            assert torch.equal(l.out_quantizer.scale, scales1[n])
            assert l.out_quantizer.frozen


def test_mismatched_settings_are_refused(tmp_path):
    """Resuming a 4-bit run into a 3-bit one would give a model that is
    neither, with nothing downstream noticing."""
    d = tmp_path / "ckpt"
    _run(_config(), ckpt_dir=d)

    other = _config()
    other.weight = QuantConfig(n_bits=3, granularity="per_channel")
    with pytest.raises(RuntimeError) as exc:
        _run(other, ckpt_dir=d)
    assert "different settings" in str(exc.value)


def test_fingerprint_tracks_the_settings_that_matter():
    a, b = _config(), _config()
    assert config_fingerprint(a) == config_fingerprint(b)

    b.weight = QuantConfig(n_bits=3, granularity="per_channel")
    assert config_fingerprint(a) != config_fingerprint(b)

    c = _config()
    c.ia_aq = IAAQConfig(enabled=False)
    assert config_fingerprint(a) != config_fingerprint(c)

    e = _config()
    e.tmas = TMASConfig(n_samples=999)
    assert config_fingerprint(a) != config_fingerprint(e)


def test_first_unfinished_stops_at_the_first_gap(tmp_path):
    """Block 5 is useless without block 4: each is calibrated on the previous
    block's quantized output."""
    ck = BlockCheckpoints(tmp_path, "fp")
    for i in (0, 1, 3):
        ck.block_path(i).write_bytes(b"x")
    assert ck.first_unfinished(8) == 2


def test_partial_write_is_not_mistaken_for_a_finished_block(tmp_path):
    ck = BlockCheckpoints(tmp_path, "fp")
    ck.block_path(0).with_suffix(".tmp").write_bytes(b"half")
    assert not ck.has_block(0)
    assert ck.first_unfinished(4) == 0
