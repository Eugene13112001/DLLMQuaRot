"""QuaRot for diffusion LMs: orthogonality, invariance, and outlier removal.

The claim rotation makes is unusually strong and therefore unusually testable:
the model's output must be *bit-for-bit equivalent up to float error*, while
its activations must be measurably flatter.  Both halves are checked here, and
the invariance half is checked across the denoising trajectory rather than at
a single state -- that is the diffusion-specific part.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from dllmquant.algos.quarot import (
    fast_hadamard,
    fuse_norm_into_linears,
    hadamard_factors,
    hadamard_matrix,
    random_hadamard_matrix,
    rotate_qk,
    rotate_residual_stream,
    rotate_value_heads,
)
from dllmquant.config import IAAQConfig, QuantConfig, RotationConfig
from dllmquant.quantizers import UniformAffineQuantizer
from dllmquant.rotate import apply_quarot, crest_factor

from test_pipeline_e2e import D, HEAD_DIM, H, VOCAB, TinyAdapter, TinyRMSNorm, _config


# ------------------------------------------------------------ construction


@pytest.mark.parametrize("n", [2, 4, 16, 64, 256, 12, 24, 96, 192])
def test_hadamard_is_orthonormal(n):
    h = hadamard_matrix(n, dtype=torch.float64)
    assert h.shape == (n, n)
    assert torch.allclose(h @ h.t(), torch.eye(n, dtype=torch.float64), atol=1e-10)


def test_hadamard_entries_are_plus_minus_one_over_sqrt_n():
    h = hadamard_matrix(64, dtype=torch.float64)
    assert torch.allclose(h.abs(), torch.full_like(h, 1 / math.sqrt(64)))


def test_factorisation_covers_real_model_widths():
    assert hadamard_factors(4096) == (1, 4096)     # LLaDA d_model
    assert hadamard_factors(12288) == (12, 1024)   # LLaDA MLP width
    assert hadamard_factors(128) == (1, 128)       # head_dim
    assert hadamard_factors(4097) is None


def test_random_hadamard_stays_orthonormal():
    h = random_hadamard_matrix(64, dtype=torch.float64, seed=3)
    assert torch.allclose(h @ h.t(), torch.eye(64, dtype=torch.float64), atol=1e-10)


def test_fast_hadamard_matches_the_explicit_matrix():
    for n in (16, 64, 96, 192):
        x = torch.randn(3, 5, n, dtype=torch.float64)
        explicit = x @ hadamard_matrix(n, dtype=torch.float64)
        assert torch.allclose(fast_hadamard(x), explicit, atol=1e-9), n


def test_fast_hadamard_preserves_norm():
    x = torch.randn(4, 1024, dtype=torch.float64)
    assert torch.allclose(fast_hadamard(x).norm(), x.norm(), atol=1e-9)


def test_no_construction_falls_back_to_orthogonal():
    with pytest.warns(RuntimeWarning):
        q = hadamard_matrix(30, dtype=torch.float64)
    assert torch.allclose(q @ q.t(), torch.eye(30, dtype=torch.float64), atol=1e-10)


# ------------------------------------------------------------- norm fusion


def test_norm_fusion_preserves_the_composition():
    torch.manual_seed(0)
    norm = TinyRMSNorm(D)
    lin = nn.Linear(D, 16, bias=False)
    x = torch.randn(4, 7, D)

    before = lin(norm(x))
    fuse_norm_into_linears(norm, [lin])
    after = lin(norm(x))

    assert torch.allclose(before, after, atol=1e-5)
    assert torch.allclose(norm.weight, torch.ones_like(norm.weight))


def test_norm_fusion_rejects_a_biased_norm():
    norm = nn.LayerNorm(D)  # has a bias
    with pytest.raises(NotImplementedError):
        fuse_norm_into_linears(norm, [nn.Linear(D, D)])


# -------------------------------------------------------------- rotations


def test_residual_rotation_leaves_the_function_unchanged():
    torch.manual_seed(0)
    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()
    ids = torch.randint(0, VOCAB - 1, (1, 10))

    before = adapter.model(ids).logits.clone()
    plan = adapter.rotation_plan()
    for norm, consumers in plan.norm_groups:
        fuse_norm_into_linears(norm, consumers)
    rotate_residual_stream(plan, random_hadamard_matrix(D, seed=1))

    assert torch.allclose(adapter.model(ids).logits, before, atol=2e-3)


@pytest.mark.parametrize("chunk", [8, 64, 4096])
def test_chunked_rotation_equals_the_whole_matrix(monkeypatch, chunk):
    """The chunking exists for memory, so it must change nothing else.

    A 157184 x 2048 embedding in float64 is 2.6 GB in and 2.6 GB out, which
    does not fit beside a model already filling the card -- but a chunk loop
    that drops or misplaces a slice would corrupt weights silently, and the
    rotation's own invariance check would then blame the rotation.
    """
    from dllmquant.algos import quarot as q

    torch.manual_seed(0)
    w = torch.randn(37, 16)  # deliberately not a multiple of any chunk
    rot = random_hadamard_matrix(16, seed=3).to(torch.float64)

    whole_right = (w.to(torch.float64) @ rot).to(w.dtype)
    whole_left = (rot.t() @ w.t().to(torch.float64)).to(w.dtype)

    monkeypatch.setattr(q, "_ROTATE_CHUNK_ELEMS", chunk)
    assert torch.equal(q._rotate_right(w, rot), whole_right)
    assert torch.equal(q._rotate_left(w.t().contiguous(), rot), whole_left)


def test_chunked_residual_rotation_is_still_invariant(monkeypatch):
    from dllmquant.algos import quarot as q

    torch.manual_seed(0)
    monkeypatch.setattr(q, "_ROTATE_CHUNK_ELEMS", 8)  # forces many chunks
    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()
    ids = torch.randint(0, VOCAB - 1, (1, 10))

    before = adapter.model(ids).logits.clone()
    plan = adapter.rotation_plan()
    for norm, consumers in plan.norm_groups:
        fuse_norm_into_linears(norm, consumers)
    rotate_residual_stream(plan, random_hadamard_matrix(D, seed=1))

    assert torch.allclose(adapter.model(ids).logits, before, atol=2e-3)


def test_value_head_rotation_leaves_the_function_unchanged():
    torch.manual_seed(0)
    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()
    ids = torch.randint(0, VOCAB - 1, (1, 10))
    before = adapter.model(ids).logits.clone()

    h = random_hadamard_matrix(HEAD_DIM, seed=2)
    for block in adapter.blocks:
        rotate_value_heads(
            block.att_proj, block.attn_out, H, HEAD_DIM, h, v_offset=2 * D
        )

    assert torch.allclose(adapter.model(ids).logits, before, atol=2e-3)


def _adapter_with_final_norm_named(*names: str):
    """A TinyAdapter whose final norm carries the given leaf name(s)."""
    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()
    original = adapter.model.ln_f
    del adapter.model.ln_f
    for name in names:
        setattr(adapter.model, name, original if name == names[0] else TinyRMSNorm(D))
    return adapter


def test_a_final_norm_called_norm_is_found():
    """LLaDA2.0 calls it `norm`; LLaDA-1.5 calls it `ln_f`.

    Not finding it is not an error at plan time -- the head simply rotates
    without its norm being fused, and the only symptom is a broken invariance
    number that looks like a bug in the rotation itself.
    """
    adapter = _adapter_with_final_norm_named("norm")
    plan = adapter.rotation_plan()

    fused = [n for n, _ in plan.norm_groups]
    assert adapter.model.norm in fused
    assert len(plan.norm_groups) == 2 * len(adapter.blocks) + 1


def test_a_specific_final_norm_wins_over_the_generic_one():
    adapter = _adapter_with_final_norm_named("ln_f", "norm")
    plan = adapter.rotation_plan()

    fused = [n for n, _ in plan.norm_groups]
    assert adapter.model.ln_f in fused
    assert adapter.model.norm not in fused


def test_a_head_with_no_final_norm_is_refused():
    from dllmquant.models.base import ArchitectureMismatch

    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()
    del adapter.model.ln_f

    with pytest.raises(ArchitectureMismatch, match="final norm"):
        adapter.rotation_plan()


def _gqa_pieces(n_heads=4, n_kv_heads=2, head_dim=4):
    d = n_heads * head_dim
    qkv = nn.Linear(d, (n_heads + 2 * n_kv_heads) * head_dim, bias=False)
    out = nn.Linear(d, d, bias=False)
    x = torch.randn(2, 5, d)

    def forward():
        b, t, _ = x.shape
        q, k, v = qkv(x).split(
            [n_heads * head_dim, n_kv_heads * head_dim, n_kv_heads * head_dim],
            dim=-1,
        )
        q = q.view(b, t, n_heads, head_dim).transpose(1, 2)
        k = k.view(b, t, n_kv_heads, head_dim).transpose(1, 2)
        v = v.view(b, t, n_kv_heads, head_dim).transpose(1, 2)
        rep = n_heads // n_kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(head_dim), dim=-1)
        return out((a @ v).transpose(1, 2).reshape(b, t, d))

    return qkv, out, forward


def test_value_rotation_survives_grouped_query_attention():
    """LLaDA2.0 has 16 query heads over 4 KV heads; V holds 4, not 16.

    Reshaping the V block as 16 heads reinterprets the matrix rather than
    rotating it -- and because slicing past the end clips instead of raising,
    it took a shape error two lines later to reveal that.
    """
    torch.manual_seed(0)
    n_heads, n_kv, hd = 4, 2, 4
    qkv, out, forward = _gqa_pieces(n_heads, n_kv, hd)

    before = forward()
    rotate_value_heads(
        qkv, out, n_heads, hd, random_hadamard_matrix(hd, seed=2),
        v_offset=(n_heads + n_kv) * hd, n_kv_heads=n_kv,
    )
    assert torch.allclose(forward(), before, atol=1e-4)


def test_a_v_block_that_overruns_the_projection_is_refused():
    """Without n_kv_heads the block is four times too tall for this weight."""
    torch.manual_seed(0)
    n_heads, n_kv, hd = 4, 2, 4
    qkv, out, _ = _gqa_pieces(n_heads, n_kv, hd)

    with pytest.raises(ValueError, match="does not fit"):
        rotate_value_heads(
            qkv, out, n_heads, hd, random_hadamard_matrix(hd, seed=2),
            v_offset=(n_heads + n_kv) * hd,
        )


def test_r4_preserves_attention_scores_exactly():
    """(QH)(KH)ᵀ = QKᵀ -- the rotation cancels inside the score matmul."""
    torch.manual_seed(4)
    q = torch.randn(2, H, 12, HEAD_DIM, dtype=torch.float64)
    k = torch.randn(2, H, 12, HEAD_DIM, dtype=torch.float64)
    before = q @ k.transpose(-1, -2)

    h = random_hadamard_matrix(HEAD_DIM, dtype=torch.float64, seed=7)
    qr, kr = rotate_qk(q, k, h)

    assert torch.allclose(qr @ kr.transpose(-1, -2), before, atol=1e-10)
    # ... and it really did change Q and K, so the test is not vacuous.
    assert not torch.allclose(qr, q)


def test_r4_flattens_q_and_k():
    torch.manual_seed(8)
    q = torch.randn(1, 2, 32, HEAD_DIM)
    q[..., 3] *= 20.0
    k = torch.randn(1, 2, 32, HEAD_DIM)

    h = random_hadamard_matrix(HEAD_DIM, seed=9)
    qr, _ = rotate_qk(q, k, h)
    assert crest_factor(qr) < crest_factor(q)


def test_r4_rejects_a_mismatched_head_dim():
    q = k = torch.randn(1, 2, 4, HEAD_DIM)
    with pytest.raises(ValueError):
        rotate_qk(q, k, random_hadamard_matrix(HEAD_DIM * 2))


def test_quarot_preset_matches_the_papers_numbers():
    from dllmquant.config import quarot_w4a4

    cfg = quarot_w4a4()
    assert cfg.activation.symmetric is True
    assert cfg.activation.clip_ratio == 0.9      # QuaRot's constant
    assert cfg.activation.granularity == "per_token"
    assert cfg.weight.mse_search is True         # squared-error clip search
    assert cfg.rotation.enabled and cfg.rotation.online_mlp


def test_clip_ratio_actually_narrows_the_range():
    x = torch.randn(8, 64)
    wide = UniformAffineQuantizer(
        QuantConfig(n_bits=4, granularity="per_token", clip_ratio=1.0)
    )
    tight = UniformAffineQuantizer(
        QuantConfig(n_bits=4, granularity="per_token", clip_ratio=0.9)
    )
    wide(x)
    tight(x)
    assert (tight.scale < wide.scale).all()


def test_rotating_only_one_side_breaks_it():
    """Guards the test above from passing for the wrong reason."""
    torch.manual_seed(0)
    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()
    ids = torch.randint(0, VOCAB - 1, (1, 10))
    before = adapter.model(ids).logits.clone()

    h = random_hadamard_matrix(HEAD_DIM, seed=2)
    block = adapter.blocks[0]
    w = block.att_proj.weight.data
    blk = w[2 * D :, :].reshape(H, HEAD_DIM, -1)
    w[2 * D :, :] = torch.einsum("ij,hjk->hik", h.t(), blk).reshape(D, -1)
    # out_proj deliberately left alone

    assert not torch.allclose(adapter.model(ids).logits, before, atol=2e-3)


# ----------------------------------------------------- end-to-end for DLLM


def _rotated_adapter(**rot_over):
    torch.manual_seed(0)
    cfg = _config()
    cfg.rotation = RotationConfig(enabled=True, **rot_over)
    adapter = TinyAdapter(cfg)
    adapter.load()
    return adapter, cfg


def test_apply_quarot_is_invariant_at_every_mask_ratio():
    adapter, cfg = _rotated_adapter()
    report = apply_quarot(adapter, cfg)

    assert set(report.invariance) == {"1.00", "0.75", "0.50", "0.25", "0.00"}
    # float32 storage: the rotation is invariant to round-off, not merely
    # "close". Anything above ~1e-6 here would mean a real defect, which is
    # exactly the sharpness bf16 cannot give.
    for ratio, rel in report.invariance.items():
        assert rel < 1e-5, f"mask ratio {ratio}: {rel:.2e}"
    assert report.tolerance == 1e-4  # resolved from dtype=float32


def test_apply_quarot_flattens_the_residual_stream():
    adapter, cfg = _rotated_adapter()
    # A randomly initialised embedding is already flat, so there would be
    # nothing to flatten. Real LLMs have persistent outlier channels -- inject
    # one, otherwise this test asserts a property of Gaussian noise.
    with torch.no_grad():
        adapter.model.wte.weight[:, 7] *= 25.0
    report = apply_quarot(adapter, cfg)

    for ratio in report.outliers_before:
        before = report.outliers_before[ratio]
        after = report.outliers_after[ratio]
        assert after <= before + 1e-6, f"mask ratio {ratio}: {before:.2f} -> {after:.2f}"


def test_the_all_mask_state_is_checked_too():
    """At mask ratio 1.0 nearly every position is the same embedding row.

    That state exists only in a diffusion LM, and it is where a missed layer
    shows up first -- so it must be part of the validation set.
    """
    adapter, cfg = _rotated_adapter()
    report = apply_quarot(adapter, cfg)
    assert "1.00" in report.invariance
    assert report.invariance["1.00"] < report.tolerance


def test_a_missed_residual_layer_is_caught():
    """Invariance must fail loudly, not silently degrade accuracy later."""
    adapter, cfg = _rotated_adapter()
    plan = adapter.rotation_plan()
    plan.output_linears.pop()  # simulate a layer the plan forgot

    for norm, consumers in plan.norm_groups:
        fuse_norm_into_linears(norm, consumers)
    ids = torch.randint(0, VOCAB - 1, (1, 10))
    before = adapter.model(ids).logits.clone()
    rotate_residual_stream(plan, random_hadamard_matrix(D, seed=1))

    with torch.no_grad():
        rel = float((adapter.model(ids).logits - before).abs().mean()
                    / before.abs().mean())
    assert rel > cfg.rotation.invariance_tol


def test_dtype_sets_the_invariance_floor():
    """bf16 keeps 7 mantissa bits, so "invariant" means something looser."""
    from dllmquant.rotate import dtype_invariance_tol

    assert dtype_invariance_tol("float32") < dtype_invariance_tol("float16")
    assert dtype_invariance_tol("float16") < dtype_invariance_tol("bfloat16")
    assert dtype_invariance_tol("something-odd") == dtype_invariance_tol("bfloat16")


def test_online_hadamard_keeps_the_original_leaf_name():
    """Renaming ff_out to `inner` would drop it out of CGQ's sequential groups,
    the skip patterns, and the value-projection search all at once."""
    from dllmquant.algos.quarot import OnlineHadamard

    adapter, _ = _rotated_adapter()
    adapter.install_online_hadamards()

    block = adapter.blocks[0]
    assert isinstance(block.ff_out, OnlineHadamard)
    leaves = {n.split(".")[-1] for n, m in block.named_modules()
              if isinstance(m, nn.Linear)}
    assert "ff_out" in leaves and "inner" not in leaves

    out = block(torch.randn(1, 6, D))
    assert out.shape == (1, 6, D)


def test_unclassified_linear_in_a_block_is_refused():
    """Structural coverage, not the numeric check, is what catches a miss."""
    from dllmquant.models.base import ArchitectureMismatch

    adapter, _ = _rotated_adapter()
    adapter.blocks[0].mystery_proj = nn.Linear(D, D, bias=False)

    with pytest.raises(ArchitectureMismatch) as exc:
        adapter.rotation_plan()
    assert "mystery_proj" in str(exc.value)


def test_rotation_plan_separates_lm_head_from_block_down_projection():
    """Both are called `ff_out` in LLaDA; only one reads the residual stream."""
    adapter, _ = _rotated_adapter()
    plan = adapter.rotation_plan()

    assert adapter.model.lm_head in plan.input_linears
    for block in adapter.blocks:
        assert block.ff_out in plan.output_linears
        assert block.ff_out not in plan.input_linears


# --------------------------------------------- what rotation buys quantizers


def test_rotation_removes_channel_outliers_for_per_token_quantization():
    """The whole point: per_token quantization is defeated by one fat channel.

    IA-AQ cannot help here -- it prioritises tokens, and this outlier lives in
    a channel, spanning every token.
    """
    torch.manual_seed(5)
    n, d = 64, 256
    x = torch.randn(n, d)
    x[:, 91] *= 40.0

    q_cfg = QuantConfig(n_bits=4, granularity="per_token")
    plain = UniformAffineQuantizer(q_cfg)
    err_plain = float((plain(x) - x).pow(2).mean())

    q = random_hadamard_matrix(d, seed=0)
    xr = x @ q
    rotated = UniformAffineQuantizer(q_cfg)
    # Undo the rotation after quantizing: this is what the fused weights do.
    err_rot = float(((rotated(xr) @ q.t()) - x).pow(2).mean())

    assert err_rot < err_plain / 2, f"rotated {err_rot:.4f} vs plain {err_plain:.4f}"
    assert crest_factor(xr) < crest_factor(x)


def test_rotation_and_ia_aq_address_different_axes():
    """Token outlier and channel outlier at once: each method fixes its own."""
    torch.manual_seed(6)
    n, d = 64, 128
    x = torch.randn(n, d)
    x[:, 17] *= 30.0   # channel outlier -> rotation's job
    x[5, :] *= 25.0    # token outlier   -> IA-AQ's job

    assert crest_factor(x @ random_hadamard_matrix(d, seed=1)) < crest_factor(x)

    # Per-feature scales already absorb the channel outlier, which is why
    # IA-AQ uses that granularity; the token outlier is what it must decide on.
    per_feature = UniformAffineQuantizer(
        QuantConfig(n_bits=4, granularity="per_feature", mse_search=False)
    )
    per_feature.find_params(x)
    assert per_feature.scale.shape == (1, d)
