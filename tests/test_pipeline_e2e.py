"""End-to-end pipeline test on a tiny synthetic DLLM.

The real checkpoints need a GPU and 16+ GB, but the pipeline's wiring does not:
this builds a 2-block LLaDA-shaped model (fused ``att_proj``, bidirectional
attention, ``ff_proj``/``ff_out`` MLP) with a handful of channels and pushes it
through TMAS -> IA-AQ -> CGQ on CPU.

What this proves: the denoising loop commits tokens and terminates, snapshots
carry a usable mask/confidence, block inputs are captured, every linear gets
wrapped and solved, the value quantizer lands on the right projection, and the
quantized model still runs.  What it cannot prove: that the *real* LLaDA
exposes these module names -- that is what scripts/selfcheck.py is for.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllmquant.config import (
    CGQConfig,
    DLLMQuantConfig,
    IAAQConfig,
    QuantConfig,
    TMASConfig,
)
from dllmquant.models.base import ModelAdapter
from dllmquant.models.llada import LLaDAAdapter, LLaDAAttentionProbe
from dllmquant.modules import QuantLinear, find_quant_linears
from dllmquant.pipeline import DLLMQuantPipeline

D, H, HEAD_DIM, VOCAB, MASK_ID = 32, 4, 8, 64, 63


class TinyRMSNorm(nn.Module):
    """RMSNorm, as in LLaDA. Not LayerNorm: mean subtraction is a projection
    that does not commute with an arbitrary rotation, so QuaRot needs this."""

    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d) * 0.1 + 1.0)

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class TinyBlock(nn.Module):
    """LLaDA-shaped: pre-norm, fused QKV, bidirectional attention, gelu MLP.

    The pre-attention norm is not decoration -- LLaDA projects the *normed*
    input, and a probe that skips it reads a different softmax than the block
    actually computed.
    """

    def __init__(self):
        super().__init__()
        self.att_proj = nn.Linear(D, 3 * D, bias=False)
        self.attn_out = nn.Linear(D, D, bias=False)
        self.ff_proj = nn.Linear(D, 2 * D, bias=False)
        self.ff_out = nn.Linear(2 * D, D, bias=False)
        self.attn_norm = TinyRMSNorm(D)
        self.ff_norm = TinyRMSNorm(D)
        self.last_attn = None  # ground truth for the probe test

    def forward(self, hidden_states, **kwargs):
        b, t, _ = hidden_states.shape
        qkv = self.att_proj(self.attn_norm(hidden_states))
        q, k, v = qkv.split(D, dim=-1)

        def heads(x):
            return x.view(b, t, H, HEAD_DIM).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        # No causal mask -- this is a diffusion LM.
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(HEAD_DIM), dim=-1)
        self.last_attn = attn.detach()
        o = (attn @ v).transpose(1, 2).reshape(b, t, D)

        h = hidden_states + self.attn_out(o)
        h = h + self.ff_out(F.gelu(self.ff_proj(self.ff_norm(h))))
        return h


class TinyOutput:
    def __init__(self, logits):
        self.logits = logits


class TinyDLLM(nn.Module):
    def __init__(self, n_blocks=2):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, D)
        self.blocks = nn.ModuleList([TinyBlock() for _ in range(n_blocks)])
        self.ln_f = TinyRMSNorm(D)
        self.lm_head = nn.Linear(D, VOCAB, bias=False)
        self.config = type("cfg", (), {
            "n_heads": H, "d_model": D, "vocab_size": VOCAB, "n_kv_heads": H,
        })()

    def forward(self, input_ids, **kwargs):
        h = self.wte(input_ids)
        for block in self.blocks:
            h = block(h)
        return TinyOutput(self.lm_head(self.ln_f(h)))


def _stable_hash(s: str) -> int:
    """FNV-1a. Python's built-in hash() for str is salted per process, which
    would make every calibration set -- and every assertion resting on it --
    differ between runs."""
    h = 2166136261
    for ch in s:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


class TinyTokenizer:
    def __call__(self, text, return_tensors=None, truncation=True, max_length=32):
        ids = [(_stable_hash(w) % (VOCAB - 2)) for w in text.split()][:max_length]
        ids = ids or [1]
        return {"input_ids": torch.tensor(ids).unsqueeze(0)}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


class TinyAdapter(ModelAdapter):
    def __init__(self, cfg):
        self.cfg = cfg
        self.mask_id = MASK_ID
        self.model = TinyDLLM()
        self.tokenizer = TinyTokenizer()
        self.n_heads, self.n_kv_heads, self.head_dim = H, H, HEAD_DIM
        self._blocks_path = "blocks"

    def load(self):
        self.model.eval()

    @property
    def blocks(self):
        return self.model.blocks

    def make_probe(self, block):
        return LLaDAAttentionProbe(block, H, H, HEAD_DIM, rotary=None)

    # The block naming mirrors LLaDA exactly, so the real plan builder applies.
    rotation_plan = LLaDAAdapter.rotation_plan


def _config(**over) -> DLLMQuantConfig:
    cfg = DLLMQuantConfig(
        model_type="llada",
        device="cpu",
        dtype="float32",
        weight=QuantConfig(n_bits=4, granularity="per_channel", mse_search=False),
        activation=QuantConfig(n_bits=4, granularity="per_token"),
        tmas=TMASConfig(
            n_samples=8, n_prompts=3, seq_len=8,
            gen_length=8, block_length=4, steps=4,
        ),
        cgq=CGQConfig(blocksize=16, act_order=True),
        ia_aq=IAAQConfig(enabled=True, n_bits=4),
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _prompts(adapter, n=3):
    return adapter.encode_prompts(
        [f"prompt number {i} with some words" for i in range(n)], max_len=8
    )


# ------------------------------------------------------------------ sampler


def test_denoising_commits_every_token():
    torch.manual_seed(0)
    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()

    prompt = _prompts(adapter, 1)[0]
    snaps = adapter.trace_generation(prompt, cfg.tmas)

    assert len(snaps) == cfg.tmas.steps
    assert snaps[0].mask_ratio == 1.0
    # Every response position must be decoded by the end.
    out = adapter.generate(prompt, cfg.tmas)
    assert (out[0, prompt.shape[-1]:] != MASK_ID).all()


def test_snapshot_confidence_is_a_probability():
    torch.manual_seed(0)
    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()
    snaps = adapter.trace_generation(_prompts(adapter, 1)[0], cfg.tmas)

    for s in snaps:
        c = s.confidence[s.mask]
        if c.numel():
            assert float(c.min()) >= 0.0 and float(c.max()) <= 1.0
        # Decoded positions carry no confidence.
        assert float(s.confidence[~s.mask].abs().max()) == 0.0


def test_mask_ratio_decreases_monotonically():
    torch.manual_seed(0)
    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()
    ratios = [s.mask_ratio for s in
              adapter.trace_generation(_prompts(adapter, 1)[0], cfg.tmas)]
    assert all(a >= b for a, b in zip(ratios, ratios[1:])), ratios


# -------------------------------------------------------------- IA-AQ probe


def test_probe_reproduces_the_blocks_own_attention():
    """The probe recomputes attention; it must match what the block computed.

    Regression for the pre-attention norm: projecting the raw residual stream
    instead of the normed input yields a plausible-looking but wrong softmax,
    and IA-AQ would then weight tokens by an attention pattern that never
    existed.
    """
    torch.manual_seed(0)
    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()

    block = adapter.blocks[0]
    ids = torch.randint(0, VOCAB - 1, (1, 12))

    probe = adapter.make_probe(block)
    with probe:
        adapter.model(ids)
        # parts are released on __exit__, so read them inside the scope.
        assert probe.parts is not None, "probe never fired"
        captured = probe.parts.attn_probs.clone()

    assert block.last_attn is not None
    assert torch.allclose(captured, block.last_attn.float(), atol=1e-5), (
        "probe attention differs from the block's own"
    )


def test_probe_finds_the_pre_attention_norm():
    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()
    probe = adapter.make_probe(adapter.blocks[0])
    assert probe.attn_norm is adapter.blocks[0].attn_norm


# ----------------------------------------------------------------- pipeline


def test_full_pipeline_quantizes_every_linear():
    torch.manual_seed(0)
    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()

    report = DLLMQuantPipeline(cfg, adapter).run(_prompts(adapter), verbose=False)

    quantized = find_quant_linears(adapter.model)
    # 4 linears per block x 2 blocks; lm_head is outside the block list.
    assert len(quantized) == 8, sorted(quantized)
    assert all(l.weight_quantized for l in quantized.values())
    assert len(report.layers) == 8
    assert report.n_calibration == cfg.tmas.n_samples


def test_ff_out_inside_a_block_is_not_skipped():
    """Regression: `ff_out` names both the MLP down-projection and the LM head."""
    cfg = _config()
    assert not cfg.skip("blocks.0.ff_out")
    assert cfg.skip("blocks.0.gate")        # MoE router stays FP
    assert not cfg.skip("blocks.0.gate_proj")  # ordinary MLP does not
    assert cfg.skip("lm_head")


def test_value_quantizer_attaches_to_the_fused_projection():
    torch.manual_seed(0)
    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()
    DLLMQuantPipeline(cfg, adapter).run(_prompts(adapter), verbose=False)

    for name, layer in find_quant_linears(adapter.model).items():
        if name.endswith("att_proj"):
            assert layer.out_quantizer is not None, "IA-AQ never attached"
            # V is the last third of a fused QKV projection.
            assert layer.out_slice == (2 * D, 3 * D)
        else:
            assert layer.out_quantizer is None


def test_quantized_model_still_generates():
    torch.manual_seed(0)
    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()
    DLLMQuantPipeline(cfg, adapter).run(_prompts(adapter), verbose=False)

    prompt = _prompts(adapter, 1)[0]
    out = adapter.generate(prompt, cfg.tmas)
    assert out.shape == (1, prompt.shape[-1] + cfg.tmas.gen_length)
    assert torch.isfinite(adapter.model(out).logits).all()


def test_pipeline_reports_ia_aq_statistics():
    torch.manual_seed(0)
    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()
    report = DLLMQuantPipeline(cfg, adapter).run(_prompts(adapter), verbose=False)

    assert set(report.ia_aq) == {"block0", "block1"}
    for stats in report.ia_aq.values():
        assert stats["weighted_mse"] >= 0
        assert "minmax_weighted_mse" in stats
        # The snapshot mask actually reached the collector, not just the config.
        assert stats["decoded_query_weight"] == 0.5


def test_certainty_weights_reach_the_solver():
    """Every dense layer must be solved with per-token weights, not uniform."""
    torch.manual_seed(0)
    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()
    report = DLLMQuantPipeline(cfg, adapter).run(_prompts(adapter), verbose=False)

    assert report.unweighted_layers == [], report.unweighted_layers
    assert all(l.weighted for l in report.layers)


def test_cgq_degrades_less_than_rtn_end_to_end():
    """CGQ vs round-to-nearest at identical numeric format.

    IA-AQ is switched off here on purpose: it quantizes the value matrix, a
    tensor the RTN baseline leaves in FP, so leaving it on would compare two
    different amounts of quantization and flatter neither method honestly.
    Its own effect is measured in ``test_ia_aq_beats_minmax_on_the_value_matrix``.
    """
    torch.manual_seed(0)
    from dllmquant.modules import wrap_linears
    from dllmquant.quantizers import quantize_weight_rtn

    cfg = _config()
    cfg.ia_aq = IAAQConfig(enabled=False)
    # A 32-channel layer at 4 bits leaves GPTQ almost no room, and the gap
    # sinks into noise. Give the solver a regime where its advantage is real:
    # 3 bits, and enough calibration tokens for a well-conditioned Hessian.
    cfg.weight = QuantConfig(n_bits=3, granularity="per_channel", mse_search=False)
    cfg.tmas = TMASConfig(
        n_samples=32, n_prompts=8, seq_len=8,
        gen_length=16, block_length=4, steps=8,
    )

    ref_adapter = TinyAdapter(cfg)
    ref_adapter.load()
    state = {k: v.clone() for k, v in ref_adapter.model.state_dict().items()}

    # Evaluate on a real mid-trajectory state, not on a fully decoded sequence.
    # CGQ deliberately optimises for masked positions -- those are the ones the
    # sampler is about to commit -- so scoring it on an all-decoded input would
    # measure it on the population it was told to de-prioritise.
    held_out = ref_adapter.trace_generation(
        ref_adapter.encode_prompts(["held out evaluation prompt text"], max_len=8)[0],
        cfg.tmas,
    )
    snap = held_out[len(held_out) // 2]
    ids = snap.input_ids.unsqueeze(0)
    scored = snap.mask.unsqueeze(0)
    assert bool(scored.any()), "held-out state has no masked positions"

    with torch.no_grad():
        fp_logits = ref_adapter.model(ids).logits.clone()

    # --- DLLMQuant ---
    a1 = TinyAdapter(cfg)
    a1.model.load_state_dict(state)
    a1.load()
    DLLMQuantPipeline(cfg, a1).run(_prompts(a1, 8), verbose=False)
    with torch.no_grad():
        err_dllm = float(
            (a1.model(ids).logits - fp_logits).pow(2)[scored].mean()
        )

    # --- RTN, same format ---
    a2 = TinyAdapter(cfg)
    a2.model.load_state_dict(state)
    a2.load()
    for bi, block in enumerate(a2.blocks):
        for layer in wrap_linears(
            block, cfg.weight, cfg.activation, skip=cfg.skip, prefix=f"blocks.{bi}"
        ).values():
            layer.set_weight(quantize_weight_rtn(layer.weight.float(), cfg.weight))
    with torch.no_grad():
        err_rtn = float(
            (a2.model(ids).logits - fp_logits).pow(2)[scored].mean()
        )

    assert err_dllm < err_rtn, f"CGQ {err_dllm:.4f} vs RTN {err_rtn:.4f}"


def test_the_two_quarot_recipes_differ_only_in_calibration_source():
    """Rotation needs no data; GPTQ does. That slot is the only difference."""
    from dllmquant.config import quarot_baseline, quarot_diffusion

    base, diff = quarot_baseline(), quarot_diffusion()

    assert base.tmas.mode == "text"
    assert diff.tmas.mode == "trajectory"
    assert diff.tmas.uniform is True  # trajectory states, but no TMAS quotas

    # Everything else is identical: same rotations, same numerics, and none of
    # the paper's three components in either.
    for cfg in (base, diff):
        assert cfg.rotation.enabled and cfg.rotation.value_heads
        assert cfg.ia_aq.enabled is False
        assert cfg.cgq.beta == 0.0  # CGQ degenerates to plain GPTQ
        assert cfg.activation.clip_ratio == 0.9


def test_quarot_baseline_recipe_runs_end_to_end():
    """The paper's QuaRot row: rotation + GPTQ + text calibration, nothing else."""
    from dllmquant.config import quarot_baseline

    torch.manual_seed(0)
    cfg = quarot_baseline()
    cfg.device, cfg.dtype = "cpu", "float32"
    cfg.weight = QuantConfig(n_bits=4, granularity="per_channel", mse_search=False)
    cfg.tmas.n_samples, cfg.tmas.seq_len = 6, 8
    cfg.rotation.online_mlp = True

    adapter = TinyAdapter(cfg)
    adapter.load()
    report = DLLMQuantPipeline(cfg, adapter).run(_prompts(adapter, 6), verbose=False)

    # No trajectory was rolled out, so nothing is masked and CGQ ran as GPTQ.
    assert cfg.tmas.mode == "text"
    assert report.n_calibration == 6
    assert not report.ia_aq, "IA-AQ must be off in the baseline"
    assert report.rotation is not None, "rotation must have run"
    assert all(l.weight_quantized for l in find_quant_linears(adapter.model).values())

    out = adapter.generate(_prompts(adapter, 1)[0], _config().tmas)
    assert torch.isfinite(adapter.model(out).logits).all()


def test_fidelity_detects_quantization_damage():
    """The cheap quality metric must move in the right direction."""
    from dllmquant.eval import ReferenceProbes, measure_fidelity

    torch.manual_seed(0)
    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()

    snaps = adapter.trace_generation(_prompts(adapter, 1)[0], cfg.tmas)
    probes = ReferenceProbes(max_positions_per_state=8).capture(
        adapter.model, snaps, device="cpu"
    )
    assert len(probes) > 0

    # Unchanged model: perfect agreement, zero divergence.
    same = measure_fidelity(adapter.model, probes, device="cpu")
    assert same.overall["top1"] == 1.0
    assert same.overall["kl"] < 1e-6

    # After quantization the metric must register damage.
    DLLMQuantPipeline(cfg, adapter).run(_prompts(adapter), verbose=False)
    after = measure_fidelity(adapter.model, probes, device="cpu")
    assert after.overall["kl"] > same.overall["kl"]
    assert after.n_positions == same.n_positions
    assert after.per_bucket, "metrics must be split by mask ratio"


def test_fidelity_reports_each_mask_ratio_separately():
    """Aggregate hides where the damage is; the split is the point."""
    from dllmquant.eval import ReferenceProbes, measure_fidelity

    torch.manual_seed(0)
    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()
    snaps = adapter.trace_generation(_prompts(adapter, 1)[0], cfg.tmas)
    probes = ReferenceProbes(max_positions_per_state=8).capture(
        adapter.model, snaps, device="cpu"
    )
    result = measure_fidelity(adapter.model, probes, device="cpu")

    assert len(result.per_bucket) >= 2
    assert sum(m["n"] for m in result.per_bucket.values()) == result.n_positions


def test_ia_aq_beats_minmax_on_the_value_matrix():
    """At the same 4-bit format, the interaction-weighted scale must not lose.

    Both numbers come from the same collected activations, so this is the fair
    head-to-head that the end-to-end comparison above cannot give.
    """
    torch.manual_seed(0)
    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.load()
    report = DLLMQuantPipeline(cfg, adapter).run(_prompts(adapter), verbose=False)

    assert report.ia_aq
    for block, stats in report.ia_aq.items():
        assert stats["weighted_mse"] <= stats["minmax_weighted_mse"] + 1e-9, (
            f"{block}: IA-AQ {stats['weighted_mse']:.6f} > "
            f"min-max {stats['minmax_weighted_mse']:.6f}"
        )


# ------------------------------------------------- naming, across the seams


def test_every_projection_name_lands_in_a_sequential_group():
    """A name that matches no group is not refused -- it is quantized last.

    LLaDA2.0 calls its projections `query_key_value` and `dense`, neither of
    which the hand-written group list contained, so attention fell through to
    the leftovers and ran *after* the MLP. The MLP was then calibrated against
    attention that was still in FP, which is the error compounding the groups
    exist to prevent, and nothing in the output said so.
    """
    from dllmquant.models.llada import (
        ATTENTION_IN_NAMES,
        ATTENTION_OUT_NAMES,
        MLP_IN_NAMES,
        MLP_OUT_NAMES,
    )
    from dllmquant.pipeline import SEQUENTIAL_GROUPS

    canonical = set(
        ATTENTION_IN_NAMES + ATTENTION_OUT_NAMES + MLP_IN_NAMES + MLP_OUT_NAMES
    )
    grouped = [name for group in SEQUENTIAL_GROUPS for name in group]

    assert canonical == set(grouped)
    assert len(grouped) == len(set(grouped)), "a name appears in two groups"


def test_the_groups_run_in_execution_order():
    from dllmquant.pipeline import SEQUENTIAL_GROUPS

    into_attention, out_of_attention, into_mlp, out_of_mlp = SEQUENTIAL_GROUPS
    assert "query_key_value" in into_attention and "att_proj" in into_attention
    assert "dense" in out_of_attention and "attn_out" in out_of_attention
    assert "gate_proj" in into_mlp and "up_proj" in into_mlp
    assert "down_proj" in out_of_mlp and "ff_out" in out_of_mlp


def test_ia_aq_attaches_to_a_fused_projection_under_gqa():
    """The V columns of a fused QKV start after Q and K, and under grouped
    query attention K is narrower than Q -- so the offset is not 2/3 of the
    width."""
    n_heads, n_kv_heads, head_dim = 16, 4, 128
    d_q, d_kv = n_heads * head_dim, n_kv_heads * head_dim

    cfg = _config()
    adapter = TinyAdapter(cfg)
    adapter.n_heads, adapter.n_kv_heads, adapter.head_dim = n_heads, n_kv_heads, head_dim

    linear = nn.Linear(2048, d_q + 2 * d_kv, bias=False)
    layer = QuantLinear(linear, cfg.weight, cfg.activation, name="attention.query_key_value")
    layers = {"blocks.1.attention.query_key_value": layer}

    pipeline = DLLMQuantPipeline(cfg, adapter)
    pipeline._attach_value_quantizer(layers, nn.Identity(), verbose=False, bi=1)

    assert layer.out_quantizer is not None, "IA-AQ never attached"
    assert layer.out_slice == (d_q + d_kv, d_q + 2 * d_kv)


def test_a_group_is_solved_in_chunks_but_only_once():
    """Every layer exactly once, no chunk over the limit, group order intact.

    One MoE block's gate/up group is 512 layers, and each holds an
    in_features^2 Hessian for the whole calibration pass -- 8.6 GB asked for
    before a single token is seen. Chunking is safe only *within* a group,
    whose members are parallel branches reading the same input; across groups
    the order carries the error compensation and must not be disturbed.
    """
    cfg = _config()
    cfg.max_group_layers = 3
    adapter = TinyAdapter(cfg)
    adapter.load()
    pipeline = DLLMQuantPipeline(cfg, adapter)

    layers = {}
    for kind, n in [("gate_proj", 7), ("down_proj", 4), ("att_proj", 1), ("attn_out", 1)]:
        for i in range(n):
            layers[f"blocks.0.mlp.experts.{i}.{kind}"] = object()

    calls = []
    pipeline._solve_group = lambda bi, block, sel, *a, **k: calls.append(list(sel))
    pipeline._run_cgq(0, None, layers, [], [], [], verbose=False)

    seen = [name for chunk in calls for name in chunk]
    assert sorted(seen) == sorted(layers), "a layer was solved twice or not at all"
    assert all(len(chunk) <= 3 for chunk in calls), "a chunk exceeded the limit"

    def kind_of(name):
        return name.split(".")[-1]

    order = [kind_of(chunk[0]) for chunk in calls]
    assert order.index("att_proj") < order.index("attn_out") < order.index("gate_proj")
    assert order.index("gate_proj") < order.index("down_proj")


def test_one_chunk_per_group_when_the_limit_is_generous():
    cfg = _config()
    cfg.max_group_layers = 1000
    adapter = TinyAdapter(cfg)
    adapter.load()
    pipeline = DLLMQuantPipeline(cfg, adapter)

    layers = {f"blocks.0.mlp.experts.{i}.gate_proj": object() for i in range(9)}
    calls = []
    pipeline._solve_group = lambda bi, block, sel, *a, **k: calls.append(list(sel))
    pipeline._run_cgq(0, None, layers, [], [], [], verbose=False)

    assert len(calls) == 1 and len(calls[0]) == 9


# ------------------------------------------- certainty weights for experts


def _reference_gather(topk_ids, expert):
    """The gather LLaDA2.0's moe_infer performs, written out.

    Kept as an independent transcription of the vendored code so the pipeline's
    reconstruction is checked against the model's actual behaviour rather than
    against itself.
    """
    k = topk_ids.shape[-1]
    idxs = topk_ids.reshape(-1).argsort()
    token_of_row = idxs // k
    expert_of_row = topk_ids.reshape(-1)[idxs]
    return token_of_row[expert_of_row == expert]


def test_expert_index_is_read_out_of_the_layer_name():
    from dllmquant.pipeline import _expert_index

    assert _expert_index("blocks.1.mlp.experts.137.gate_proj") == 137
    assert _expert_index("blocks.1.mlp.shared_experts.up_proj") is None
    assert _expert_index("blocks.1.attention.query_key_value") is None


def test_weights_follow_the_rows_the_router_actually_sent():
    """Certainty weighting is the diffusion-specific half of CGQ, and on an
    MoE it was reaching 3% of the layers: an expert sees a gathered subset, the
    token count stops matching, and the weights were dropped. Order matters as
    much as membership -- the same weights in the wrong order would reweight
    one token's activations by another token's certainty."""
    from dllmquant.pipeline import _weights_for_expert

    torch.manual_seed(0)
    n_tokens, k, n_experts = 12, 2, 4
    routes = torch.randint(0, n_experts, (n_tokens, k))
    weights = torch.arange(n_tokens, dtype=torch.float32)

    for expert in range(n_experts):
        expected_tokens = _reference_gather(routes, expert)
        got, reason = _weights_for_expert(
            f"blocks.1.mlp.experts.{expert}.gate_proj",
            weights, routes, len(expected_tokens),
        )
        if len(expected_tokens) == 0:
            continue
        assert got is not None, f"expert {expert} got no weights: {reason}"
        assert torch.equal(got, weights[expected_tokens])
        assert reason == ""


def test_a_mismatch_falls_back_instead_of_guessing():
    from dllmquant.pipeline import _weights_for_expert

    routes = torch.randint(0, 4, (12, 2))
    weights = torch.arange(12, dtype=torch.float32)

    # A row count that cannot be right for this expert. The reason is part of
    # the contract: a run that falls back on every expert has to be able to say
    # which of the four ways it failed, or the next fix is a guess.
    got, reason = _weights_for_expert(
        "blocks.1.mlp.experts.0.gate_proj", weights, routes, n_rows=999
    )
    assert got is None and "layer saw 999" in reason

    assert _weights_for_expert("blocks.1.mlp.down_proj", weights, routes, 6) == (
        None, "not an expert layer"
    )
    assert _weights_for_expert(
        "blocks.1.mlp.experts.0.gate_proj", weights, None, 6
    ) == (None, "router output never seen")
    assert _weights_for_expert(
        "blocks.1.attention.dense", weights, routes, n_rows=12
    ) == (None, "not an expert layer")


def test_the_router_watcher_captures_the_discrete_choice():
    from dllmquant.pipeline import _watch_routers

    class Gate(nn.Module):
        num_experts = 4
        top_k = 2

        def forward(self, x):
            idx = torch.tensor([[0, 1], [2, 3]])
            return idx, torch.ones_like(idx, dtype=torch.float), torch.zeros(2, 4)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = Gate()

        def forward(self, x):
            return self.gate(x)

    block, state = Block(), {"routes": None}
    handles = _watch_routers(block, state)
    try:
        block(torch.zeros(2, 4))
    finally:
        for h in handles:
            h.remove()

    assert state["routes"] is not None
    assert state["routes"].tolist() == [[0, 1], [2, 3]]


def test_weights_are_indexed_by_token_not_by_batch():
    """Certainty weights arrive shaped like the snapshot, [B, S].

    `chosen` holds flat token indices, so indexing a 2-D tensor with them
    walks the batch dimension instead -- and the bound check compares against
    numel(), which waves that straight through. It surfaced as a CUDA
    device-side assert several calls downstream; on CPU it would have been a
    quietly wrong Hessian. It stayed hidden this long only because the row
    counts disagreed first and the fallback fired before the indexing.
    """
    from dllmquant.pipeline import _weights_for_expert

    torch.manual_seed(0)
    n_tokens, k, n_experts = 12, 2, 4
    routes = torch.randint(0, n_experts, (n_tokens, k))
    flat = torch.arange(n_tokens, dtype=torch.float32)
    shaped = flat.reshape(1, n_tokens)          # what the snapshot hands over

    for expert in range(n_experts):
        expected = _reference_gather(routes, expert)
        if len(expected) == 0:
            continue
        got_flat, _ = _weights_for_expert(
            f"b.mlp.experts.{expert}.gate_proj", flat, routes, len(expected))
        got_shaped, reason = _weights_for_expert(
            f"b.mlp.experts.{expert}.gate_proj", shaped, routes, len(expected))
        assert got_shaped is not None, reason
        assert torch.equal(got_shaped, got_flat)
        assert torch.equal(got_shaped, flat[expected])


def test_max_blocks_stops_early_and_says_so(capsys):
    """A question about a 20-hour run should not cost 20 hours to ask."""
    torch.manual_seed(0)
    cfg = _config(max_blocks=1)
    adapter = TinyAdapter(cfg)
    adapter.load()

    report = DLLMQuantPipeline(cfg, adapter).run(_prompts(adapter), verbose=True)

    solved = {l.name.split(".")[1] for l in report.layers}
    assert solved == {"0"}, f"expected block 0 only, got {sorted(solved)}"
    assert "half quantized" in capsys.readouterr().out

    # And the default must still do every block, or this flag is a foot-gun.
    torch.manual_seed(0)
    adapter = TinyAdapter(_config())
    adapter.load()
    full = DLLMQuantPipeline(_config(), adapter).run(_prompts(adapter), verbose=False)
    assert {l.name.split(".")[1] for l in full.layers} == {"0", "1"}
