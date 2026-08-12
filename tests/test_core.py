"""CPU unit tests for the DLLMQuant core.

None of these need a GPU or model weights -- they check the algorithms on
small synthetic tensors.  Run with:  python -m pytest tests -q
"""

from __future__ import annotations

import pytest
import torch

from dllmquant.algos.cgq import CGQ, certainty_weights
from dllmquant.algos.ia_aq import InteractionCollector, interaction_weights
from dllmquant.calib.tmas import (
    Snapshot,
    TMASSampler,
    allocate,
    bucket_index,
)
from dllmquant.config import CGQConfig, IAAQConfig, QuantConfig, TMASConfig
from dllmquant.modules import QuantLinear, wrap_linears
from dllmquant.quantizers import UniformAffineQuantizer, quantize_weight_rtn

torch.manual_seed(0)


# ---------------------------------------------------------------- quantizers


def test_quantized_values_land_on_the_grid():
    x = torch.randn(32, 64)
    q = UniformAffineQuantizer(QuantConfig(n_bits=4, granularity="per_channel"))
    q.find_params(x)
    x_hat = q(x)

    # Recover integer codes; they must be integers inside [qmin, qmax].
    codes = (x_hat / q.scale) + q.zero_point
    assert torch.allclose(codes, codes.round(), atol=1e-4)
    assert codes.min() >= q.qmin - 1e-4
    assert codes.max() <= q.qmax + 1e-4
    assert len(torch.unique(codes.round())) <= 2**4


def test_error_is_bounded_by_half_a_step():
    x = torch.randn(16, 128)
    q = UniformAffineQuantizer(QuantConfig(n_bits=4, granularity="per_channel"))
    q.find_params(x)
    err = (q(x) - x).abs()
    assert (err <= q.scale / 2 + 1e-5).all()


def test_quantizer_preserves_dtype():
    """A fake quantizer must be transparent in dtype, always."""
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(8, 32, dtype=dtype)
        for gran in ("per_token", "per_channel", "per_feature", "per_tensor"):
            q = UniformAffineQuantizer(QuantConfig(n_bits=4, granularity=gran))
            assert q(x).dtype == dtype, (dtype, gran)


def test_fp32_calibrated_scale_does_not_promote_a_bf16_tensor():
    """IA-AQ calibrates in fp32; applying it must not upcast the activation.

    Regression: a promoted value projection breaks fused attention, because q
    and k stay bf16 while v turns fp32 and SDPA refuses the mix.
    """
    from dllmquant.quantizers import InteractionAwareQuantizer

    calib = torch.randn(256, 32, dtype=torch.float32)
    q = InteractionAwareQuantizer(
        QuantConfig(n_bits=4, granularity="per_feature", mse_search=True)
    )
    q.find_params(calib, weights=torch.rand(256))
    q.freeze()
    assert q.scale.dtype == torch.float32

    x = torch.randn(4, 16, 32, dtype=torch.bfloat16)
    out = q(x)
    assert out.dtype == torch.bfloat16
    assert out.shape == x.shape


def test_quant_linear_preserves_dtype_with_an_output_quantizer():
    from dllmquant.quantizers import InteractionAwareQuantizer

    linear = torch.nn.Linear(32, 96, bias=False).to(torch.bfloat16)
    ql = QuantLinear(
        linear,
        QuantConfig(n_bits=4, granularity="per_channel"),
        QuantConfig(n_bits=4, granularity="per_token"),
    )
    vq = InteractionAwareQuantizer(
        QuantConfig(n_bits=4, granularity="per_feature", mse_search=False)
    )
    vq.find_params(torch.randn(128, 32, dtype=torch.float32))
    vq.freeze()
    ql.out_quantizer = vq
    ql.out_slice = (64, 96)  # V slice of a fused QKV projection

    out = ql(torch.randn(2, 5, 32, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16
    assert out.shape == (2, 5, 96)


def test_16_bits_is_a_no_op():
    x = torch.randn(8, 16)
    q = UniformAffineQuantizer(QuantConfig(n_bits=16, granularity="per_token"))
    assert torch.equal(q(x), x)


def test_more_bits_never_hurt():
    x = torch.randn(64, 128)
    errs = []
    for bits in (2, 3, 4, 8):
        cfg = QuantConfig(n_bits=bits, granularity="per_channel")
        q = UniformAffineQuantizer(cfg)
        q.find_params(x)
        errs.append(float((q(x) - x).pow(2).mean()))
    assert errs == sorted(errs, reverse=True), errs


def test_per_feature_shares_one_scale_across_tokens():
    x = torch.randn(100, 32)
    q = UniformAffineQuantizer(QuantConfig(n_bits=4, granularity="per_feature"))
    q.find_params(x)
    assert q.scale.shape == (1, 32)


def test_per_token_is_dynamic():
    q = UniformAffineQuantizer(QuantConfig(n_bits=4, granularity="per_token"))
    small = torch.randn(4, 16) * 0.01
    q(small)
    scale_small = q.scale.clone()
    q(torch.randn(4, 16) * 100.0)
    assert (q.scale > scale_small).all()


def test_dynamic_group_activations_track_a_changing_batch():
    """per_group activations need cfg.dynamic, or they freeze the first batch.

    This is the cheapest mitigation available here for channel-wise outliers:
    one scale per (token, group of channels) instead of one per token.
    """
    cfg = QuantConfig(n_bits=4, granularity="per_group", group_size=8, dynamic=True)
    q = UniformAffineQuantizer(cfg)

    q(torch.randn(2, 4, 32) * 0.01)
    small = q.scale.clone()
    out = q(torch.randn(5, 9, 32) * 100.0)  # different shape AND magnitude

    assert out.shape == (5, 9, 32)          # a frozen scale would not broadcast
    assert q.scale.shape[:2] == (5, 9)      # one scale per token per group
    assert float(q.scale.mean()) > float(small.mean())


def test_per_group_activations_beat_per_token_on_channel_outliers():
    """One fat channel ruins the grid for the whole token under per_token."""
    torch.manual_seed(11)
    x = torch.randn(16, 256)
    x[:, 37] *= 60.0  # a persistent outlier channel, as seen in real LLMs

    per_token = UniformAffineQuantizer(
        QuantConfig(n_bits=4, granularity="per_token")
    )
    per_group = UniformAffineQuantizer(
        QuantConfig(n_bits=4, granularity="per_group", group_size=32, dynamic=True)
    )
    e_token = float((per_token(x) - x).pow(2).mean())
    e_group = float((per_group(x) - x).pow(2).mean())
    assert e_group < e_token, f"per_group {e_group:.4f} vs per_token {e_token:.4f}"


def test_group_quantization_beats_per_tensor():
    # Channels with wildly different scales are exactly what grouping fixes.
    x = torch.randn(64, 128) * torch.logspace(-2, 2, 128).unsqueeze(0)
    per_tensor = QuantConfig(n_bits=4, granularity="per_tensor")
    per_group = QuantConfig(n_bits=4, granularity="per_group", group_size=32)

    e_tensor = float((quantize_weight_rtn(x, per_tensor) - x).pow(2).mean())
    e_group = float((quantize_weight_rtn(x, per_group) - x).pow(2).mean())
    assert e_group < e_tensor


# ---------------------------------------------------------------------- TMAS


def test_allocate_sums_exactly():
    for n in (1, 7, 128, 1000):
        counts = allocate(n, [0.3, 0.2, 0.2, 0.3])
        assert sum(counts) == n
        assert all(c >= 0 for c in counts)


def test_allocate_follows_the_u_shape():
    counts = allocate(100, [0.3, 0.2, 0.2, 0.3])
    assert counts == [30, 20, 20, 30]
    # The extremes get more budget than the middle.
    assert counts[0] > counts[1] and counts[3] > counts[2]


def test_bucket_index_orders_by_mask_ratio():
    # ratio 1.0 = fully masked = earliest = bucket 0
    assert bucket_index(1.0, 4) == 0
    assert bucket_index(0.9, 4) == 0
    assert bucket_index(0.6, 4) == 1
    assert bucket_index(0.3, 4) == 2
    assert bucket_index(0.0, 4) == 3
    assert bucket_index(-5.0, 4) == 3  # clamped
    assert bucket_index(5.0, 4) == 0


def _fake_snapshot(mask_ratio: float, step: int = 0) -> Snapshot:
    L = 16
    n_masked = int(round(mask_ratio * L))
    mask = torch.zeros(L, dtype=torch.bool)
    mask[:n_masked] = True
    return Snapshot(
        input_ids=torch.zeros(L, dtype=torch.long),
        mask=mask,
        confidence=torch.rand(L),
        step=step,
        total_steps=64,
        block_idx=0,
        mask_ratio=mask_ratio,
    )


def test_tmas_selection_respects_proportions():
    # A uniform trajectory: equally many snapshots at every mask ratio.
    snaps = [_fake_snapshot(r / 200.0, r) for r in range(201)] * 4
    cfg = TMASConfig(n_samples=100, n_buckets=4, proportions=(0.3, 0.2, 0.2, 0.3))
    sampler = TMASSampler(cfg)
    chosen = sampler.select(snaps)

    assert len(chosen) == 100
    counts = [0] * 4
    for s in chosen:
        counts[bucket_index(s.mask_ratio, 4)] += 1
    assert counts == [30, 20, 20, 30]


def test_text_calibration_has_no_masked_positions():
    """The baseline path: no trajectory, so CGQ must degenerate to GPTQ."""
    from dllmquant.calib.tmas import text_calibration_set

    prompts = [torch.arange(20) for _ in range(6)]
    cfg = TMASConfig(n_samples=4, seq_len=12)
    snaps = text_calibration_set(prompts, cfg, verbose=False)

    assert len(snaps) == 4
    for s in snaps:
        assert s.input_ids.numel() == 12
        assert not s.mask.any()
        assert s.mask_ratio == 0.0

    # With nothing masked, certainty weights come out uniform on their own --
    # no special-casing needed for the baseline to be a true GPTQ.
    c = certainty_weights(snaps[0].mask, snaps[0].confidence, CGQConfig())
    assert torch.allclose(c, torch.ones_like(c))


def test_text_mode_is_validated():
    try:
        TMASConfig(mode="whatever")
    except ValueError as e:
        assert "trajectory" in str(e)
    else:
        raise AssertionError("expected a ValueError for an unknown mode")


def test_uniform_ablation_inherits_the_trajectory_distribution():
    """`uniform=True` must reproduce naive sampling, or TMAS has no control.

    Without this switch the first of the paper's three components cannot be
    ablated: there would be nothing to compare the U-shaped allocation against.
    """
    torch.manual_seed(3)
    middle = [_fake_snapshot(0.25 + 0.5 * torch.rand(1).item()) for _ in range(320)]
    extremes = [_fake_snapshot(torch.rand(1).item() * 0.25) for _ in range(40)]
    extremes += [_fake_snapshot(0.75 + 0.25 * torch.rand(1).item()) for _ in range(40)]
    snaps = middle + extremes

    cfg = TMASConfig(n_samples=200, proportions=(0.3, 0.2, 0.2, 0.3), uniform=True)
    chosen = TMASSampler(cfg).select(snaps)
    assert len(chosen) == 200

    counts = [0] * 4
    for s in chosen:
        counts[bucket_index(s.mask_ratio, 4)] += 1
    share = [c / 200 for c in counts]

    # Uniform sampling keeps the trajectory's own middle-heavy shape ...
    assert share[1] + share[2] > 0.6
    # ... unlike the quota-based path, which would force exactly 0.3/0.2/0.2/0.3.
    assert abs(share[0] - 0.3) > 0.05


def test_tmas_redistributes_when_a_bucket_is_empty():
    # No snapshot ever lands in the most-masked bucket.
    snaps = [_fake_snapshot(r / 400.0, r) for r in range(300)]
    cfg = TMASConfig(n_samples=80, n_buckets=4, proportions=(0.3, 0.2, 0.2, 0.3))
    chosen = TMASSampler(cfg).select(snaps)
    assert len(chosen) == 80


def test_tmas_corrects_a_middle_heavy_trajectory():
    """The reason TMAS exists: naive sampling inherits the trajectory's bias.

    Block-wise decoding spends most of its steps at intermediate mask ratios,
    so a calibration set drawn uniformly over steps under-represents exactly
    the two extremes where the token distribution is most unusual.
    """
    torch.manual_seed(7)
    # 80% of steps sit in the middle two buckets, 20% at the extremes.
    middle = [_fake_snapshot(0.25 + 0.5 * torch.rand(1).item()) for _ in range(320)]
    extremes = [_fake_snapshot(torch.rand(1).item() * 0.25) for _ in range(40)]
    extremes += [_fake_snapshot(0.75 + 0.25 * torch.rand(1).item()) for _ in range(40)]
    snaps = middle + extremes

    sampler = TMASSampler(TMASConfig(n_samples=100, proportions=(0.3, 0.2, 0.2, 0.3)))
    natural = [len(b) / len(snaps) for b in sampler.bucketize(snaps)]

    counts = [0] * 4
    for s in sampler.select(snaps):
        counts[bucket_index(s.mask_ratio, 4)] += 1
    tmas = [c / 100 for c in counts]

    # Uniform sampling would starve both extremes; TMAS hits its quotas.
    assert natural[0] < 0.2 and natural[3] < 0.2
    assert tmas == [0.3, 0.2, 0.2, 0.3]


# ----------------------------------------------------------------------- CGQ


def test_certainty_weights_prioritise_confident_masked_tokens():
    mask = torch.tensor([True, True, False, False])
    conf = torch.tensor([0.95, 0.10, 0.99, 0.99])
    cfg = CGQConfig(unmasked_weight=0.5, masked_base=0.5, beta=1.0, normalize=False)
    c = certainty_weights(mask, conf, cfg)

    assert c[0] > c[1]        # confident masked > unconfident masked
    assert c[0] > c[2]        # confident masked > already decoded
    assert c[2] == c[3] == 0.5  # decoded tokens ignore their (stale) confidence


def test_certainty_weights_normalise_to_mean_one():
    mask = torch.rand(256) > 0.5
    conf = torch.rand(256)
    c = certainty_weights(mask, conf, CGQConfig(normalize=True))
    assert abs(float(c.mean()) - 1.0) < 1e-5


def _correlated_inputs(n: int, d: int) -> torch.Tensor:
    """Calibration data with a non-diagonal covariance, where GPTQ has an edge."""
    a = torch.randn(d, d) / (d**0.5)
    return torch.randn(n, d) @ a


def test_cgq_beats_rtn_on_output_error():
    torch.manual_seed(1)
    d_in, d_out, n = 128, 64, 2048
    linear = torch.nn.Linear(d_in, d_out, bias=False)
    x = _correlated_inputs(n, d_in)

    w_cfg = QuantConfig(n_bits=3, granularity="per_channel")
    a_cfg = QuantConfig(n_bits=16, granularity="per_token")
    layer = QuantLinear(linear, w_cfg, a_cfg, name="test")
    w_fp = layer.weight.clone()

    solver = CGQ(layer, CGQConfig(act_order=True), w_cfg)
    solver.add_batch(x)
    w_cgq, _ = solver.quantize()
    w_rtn = quantize_weight_rtn(w_fp, w_cfg)

    ref = x @ w_fp.t()
    err_cgq = float((x @ w_cgq.t().float() - ref).pow(2).mean())
    err_rtn = float((x @ w_rtn.t() - ref).pow(2).mean())
    assert err_cgq < err_rtn, f"CGQ {err_cgq:.5f} vs RTN {err_rtn:.5f}"


def test_certainty_weighting_helps_on_the_tokens_it_prioritises():
    """Down-weighted tokens should lose accuracy to the up-weighted ones."""
    torch.manual_seed(2)
    d_in, d_out, n = 96, 48, 1536
    linear = torch.nn.Linear(d_in, d_out, bias=False)
    w_cfg = QuantConfig(n_bits=3, granularity="per_channel")
    a_cfg = QuantConfig(n_bits=16, granularity="per_token")

    # Two token populations with different covariance: "masked" and "decoded".
    x_masked = _correlated_inputs(n // 2, d_in)
    x_decoded = _correlated_inputs(n // 2, d_in) * 1.5
    x = torch.cat([x_masked, x_decoded], dim=0)

    mask = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)]).bool()
    conf = torch.cat([torch.full((n // 2,), 0.9), torch.full((n // 2,), 0.9)])

    def run(weights):
        layer = QuantLinear(linear, w_cfg, a_cfg, name="t")
        solver = CGQ(layer, CGQConfig(act_order=True), w_cfg)
        solver.add_batch(x, weights)
        q, _ = solver.quantize()
        return q

    w_fp = linear.weight.data.clone()
    cgq_cfg = CGQConfig(unmasked_weight=0.1, masked_base=0.5, beta=1.0)
    c = certainty_weights(mask, conf, cgq_cfg)

    q_weighted = run(c)
    q_plain = run(None)

    def err_on(q, xs):
        return float((xs @ q.t().float() - xs @ w_fp.t()).pow(2).mean())

    # On the prioritised (masked) population the weighted solve must win.
    assert err_on(q_weighted, x_masked) < err_on(q_plain, x_masked)


def _starved_solver(n_bits=4):
    linear = torch.nn.Linear(8, 8, bias=False)
    w_cfg = QuantConfig(n_bits=n_bits, granularity="per_channel")
    layer = QuantLinear(linear, w_cfg, QuantConfig(n_bits=16, granularity="per_token"))
    return layer, CGQ(layer, CGQConfig(), w_cfg)


def test_a_layer_with_no_calibration_falls_back_to_rounding():
    """In a 256-expert MoE most experts are reached by nothing at all.

    Refusing to quantize them means abandoning an hours-long run over a weight
    that plain rounding handles about as well as anything could without data.
    """
    layer, solver = _starved_solver()
    q, loss = solver.quantize()

    assert solver.starved is True
    assert loss == 0.0
    assert q.shape == layer.weight.shape
    assert torch.isfinite(q).all()
    # It really was quantized, not passed through.
    assert not torch.allclose(q, layer.weight.data.float())
    assert len(q.unique()) <= 8 * 2**4  # per-row scales, 4 bits each


def test_the_fallback_is_not_silent():
    """A zero proxy loss on a starved layer means 'nothing to measure', not
    'no error' -- so the flag, not the loss, is what callers must read."""
    _, solver = _starved_solver()
    assert solver.starved is False
    solver.quantize()
    assert solver.starved is True


def test_a_solved_layer_is_not_marked_starved():
    layer, solver = _starved_solver()
    solver.add_batch(torch.randn(64, 8))
    solver.quantize()
    assert solver.starved is False


# ---------------------------------------------------------------------- IA-AQ


def test_interaction_weights_follow_attention_mass():
    b, h, q_len, k_len = 1, 4, 8, 8
    attn = torch.zeros(b, h, q_len, k_len)
    attn[..., 3] = 1.0  # every query attends only to token 3
    w = interaction_weights(attn, IAAQConfig())

    assert w.shape == (b, k_len)
    assert int(w.argmax(dim=-1)) == 3
    assert abs(float(w.mean()) - 1.0) < 1e-4


def _split_attention(n: int = 8, cut: int = 4):
    """Masked queries all look at token 2, decoded queries all at token 6."""
    attn = torch.zeros(1, 2, n, n)
    attn[:, :, :cut, 2] = 1.0
    attn[:, :, cut:, 6] = 1.0
    q_mask = torch.zeros(1, n, dtype=torch.bool)
    q_mask[0, :cut] = True
    return attn, q_mask


def test_decoded_queries_are_down_weighted_not_dropped():
    """A decoded query's output is discarded only at the FINAL layer.

    Below it, that output becomes the hidden state which -- one layer up --
    serves as key and value for the masked positions. The influence is
    indirect and attenuated, so the row is scaled, not removed.
    """
    attn, q_mask = _split_attention()

    w_full = interaction_weights(attn, IAAQConfig(decoded_query_weight=1.0), q_mask)
    w_soft = interaction_weights(attn, IAAQConfig(decoded_query_weight=0.5), q_mask)

    # Counting decoded rows fully, both tokens look equally important.
    assert abs(float(w_full[0, 2]) - float(w_full[0, 6])) < 1e-4
    # Down-weighted, token 2 leads -- but token 6 keeps real weight.
    assert float(w_soft[0, 2]) > float(w_soft[0, 6])
    assert float(w_soft[0, 6]) > 0.3 * float(w_soft[0, 2])


def test_zero_weight_reproduces_hard_exclusion():
    attn, q_mask = _split_attention()
    w = interaction_weights(attn, IAAQConfig(decoded_query_weight=0.0), q_mask)
    assert float(w[0, 2]) > float(w[0, 6]) * 100


def test_weight_of_one_makes_the_mask_a_no_op():
    """Then IA-AQ behaves exactly as it would on an autoregressive model."""
    attn, q_mask = _split_attention()
    v = torch.randn(1, 8, 8)

    off = InteractionCollector(IAAQConfig(decoded_query_weight=1.0))
    off.add(v, attn, q_mask)
    assert off.used_query_mask is False

    on = InteractionCollector(IAAQConfig(decoded_query_weight=0.5))
    on.add(v, attn, q_mask)
    assert on.used_query_mask is True
    assert not torch.allclose(off.weights[0], on.weights[0])


def test_all_decoded_state_still_ranks_tokens():
    """The trajectory's final state has nothing masked; every row is scaled by
    the same factor, so the ranking must survive rather than collapse."""
    v = torch.randn(1, 8, 16)
    attn = torch.softmax(torch.randn(1, 2, 8, 8), dim=-1)
    empty = torch.zeros(1, 8, dtype=torch.bool)

    collector = InteractionCollector(IAAQConfig(n_bits=4))
    collector.add(v, attn, empty)

    w = collector.weights[0]
    assert torch.isfinite(w).all()
    assert float(w.std()) > 0, "weights collapsed to a constant"
    assert abs(float(w.mean()) - 1.0) < 1e-4


def test_interaction_weights_reject_bad_shapes():
    try:
        interaction_weights(torch.rand(4, 8, 8), IAAQConfig())
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError for a 3-D attention tensor")


def test_ia_aq_protects_high_attention_tokens():
    """Outliers on ignored tokens must not set the step size for read tokens."""
    torch.manual_seed(3)
    b, h, n, d = 1, 2, 128, 32

    v = torch.randn(b, n, d)
    v[:, 64:, :] *= 50.0          # huge magnitudes on tokens nobody reads

    attn = torch.rand(b, h, n, n) * 1e-6
    attn[..., :64] = 1.0          # all attention mass on the first half
    attn = attn / attn.sum(dim=-1, keepdim=True)

    collector = InteractionCollector(IAAQConfig(n_bits=4))
    collector.add(v, attn)
    q_ia = collector.build_quantizer()

    # Baseline: same format, plain min/max over everything.
    q_plain = UniformAffineQuantizer(
        QuantConfig(n_bits=4, granularity="per_feature", mse_search=False)
    )
    q_plain.find_params(v.reshape(-1, d))

    important = v[:, :64, :].reshape(-1, d)
    err_ia = float((q_ia(important) - important).pow(2).mean())
    err_plain = float((q_plain(important) - important).pow(2).mean())
    assert err_ia < err_plain, f"IA-AQ {err_ia:.4f} vs min-max {err_plain:.4f}"


def test_ia_aq_quantizer_is_frozen_after_calibration():
    v = torch.randn(1, 64, 16)
    attn = torch.softmax(torch.randn(1, 2, 64, 64), dim=-1)
    collector = InteractionCollector(IAAQConfig(n_bits=4))
    collector.add(v, attn)
    q = collector.build_quantizer()

    scale_before = q.scale.clone()
    q(torch.randn(1, 64, 16) * 1000.0)  # a wild batch must not move the scale
    assert torch.equal(q.scale, scale_before)


# -------------------------------------------------------------- module wiring


def test_wrap_linears_replaces_and_skips():
    net = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 8),
    )
    w_cfg = QuantConfig(n_bits=4, granularity="per_channel")
    a_cfg = QuantConfig(n_bits=4, granularity="per_token")
    replaced = wrap_linears(net, w_cfg, a_cfg, skip=lambda n: n == "2")

    assert set(replaced) == {"0"}
    assert isinstance(net[0], QuantLinear)
    assert isinstance(net[2], torch.nn.Linear)

    out = net(torch.randn(4, 16))
    assert out.shape == (4, 8)


def test_wrapping_frees_each_linear_before_taking_the_next():
    """Peak memory, measured by reference rather than by megabytes.

    The cost is paid *during* the walk, not after it: holding the list of
    targets keeps every superseded weight resident until the loop ends, which
    doubles a block's footprint at the peak -- about 1.6 GB for one LLaDA2.0
    MoE block, enough to lose a run on a shared card.  Checking after the call
    proves nothing, since the list dies with the function either way, so the
    skip callback is used as a window into the middle of the loop.
    """
    import gc
    import weakref

    net = torch.nn.Sequential(*[torch.nn.Linear(16, 16) for _ in range(4)])
    ghosts = [weakref.ref(m) for m in net]
    already_freed = []

    def probe(name: str) -> bool:
        gc.collect()
        already_freed.append(sum(g() is None for g in ghosts))
        return False

    w_cfg = QuantConfig(n_bits=4, granularity="per_channel")
    a_cfg = QuantConfig(n_bits=4, granularity="per_token")
    wrap_linears(net, w_cfg, a_cfg, skip=probe)

    # Entering the n-th iteration, the n-1 already replaced must be gone.
    assert already_freed == [0, 1, 2, 3]


def test_quant_linear_matches_fp_when_quantization_is_off():
    linear = torch.nn.Linear(16, 8)
    q = QuantLinear(
        linear,
        QuantConfig(n_bits=16, granularity="per_channel"),
        QuantConfig(n_bits=16, granularity="per_token"),
    )
    x = torch.randn(4, 16)
    assert torch.allclose(q(x), linear(x), atol=1e-6)


def test_collect_inputs_captures_post_activation_quant():
    linear = torch.nn.Linear(16, 8)
    q = QuantLinear(
        linear,
        QuantConfig(n_bits=4, granularity="per_channel"),
        QuantConfig(n_bits=4, granularity="per_token"),
    )
    x = torch.randn(4, 16)
    with q.collect_inputs() as buf:
        q(x)
        assert len(buf) == 1
        captured = buf[0]
    # What the matmul saw is the quantized input, not the raw one.
    assert not torch.allclose(captured, x)
    assert captured.shape == x.shape


# ------------------------------------------------------- preflight budgeting


def test_a_device_map_budget_is_the_sum_over_cards(monkeypatch):
    """Checking card 0 alone refused a job with 59 GB across two cards.

    With a device map the model is spread over every visible GPU, so the
    question is not "does card 0 hold it" -- the guard was right about the
    number and wrong about the question.
    """
    from dllmquant.models.base import preflight_memory

    free = {0: 29.4 * 2**30, 1: 29.6 * 2**30}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i=0: (free[i], 140 * 2**30))

    assert preflight_memory(34.0, strict=True, device_map="auto") > 34.0

    with pytest.raises(RuntimeError, match="only"):
        preflight_memory(34.0, strict=True)


def test_a_single_card_job_still_checks_that_card(monkeypatch):
    from dllmquant.models.base import preflight_memory

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda i=0: (40 * 2**30, 140 * 2**30)
    )

    assert preflight_memory(34.0, strict=True) == pytest.approx(40.0)
