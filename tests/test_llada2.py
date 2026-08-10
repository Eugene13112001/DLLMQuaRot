"""What LLaDA2.0 needs that LLaDA-1.5 does not.

Two models now share one pipeline, and the failure mode worth guarding against
is not a crash -- it is the block-diffusion model quietly being fed LLaDA-1.5's
assumptions (full attention, blocks anchored at the prompt), or the reverse,
which would invalidate the LLaDA-1.5 numbers already measured.

Nothing here loads a checkpoint: the mask and the block layout are arithmetic,
and arithmetic is exactly what got them wrong in the first place.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from dllmquant.config import DLLMQuantConfig, TMASConfig
from dllmquant.models.base import ModelAdapter, discover_blocks
from dllmquant.models.llada2_moe import LLaDA2MoEAdapter


# --------------------------------------------------------------- block finder


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)


class _Layer(nn.Module):
    """A decoder layer: has attention, and owns a long list of experts."""

    def __init__(self, n_experts: int):
        super().__init__()
        self.self_attn = _Attn()
        self.experts = nn.ModuleList(
            [nn.Linear(4, 4, bias=False) for _ in range(n_experts)]
        )


class _MoE(nn.Module):
    def __init__(self, n_layers=3, n_experts=16):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(n_experts) for _ in range(n_layers)])


def test_expert_list_is_not_mistaken_for_the_layer_stack():
    """The trap that made LLaDA2.0-mini report 'blocks at layers.1.mlp.experts'.

    256 experts against 20 layers: picking the longest uniform ModuleList picks
    the experts of a single layer, and every later stage -- rotation, CGQ,
    the attention probe -- then works on a model nobody meant to touch.
    """
    name, blocks = discover_blocks(_MoE(n_layers=3, n_experts=16))
    assert name == "layers"
    assert len(blocks) == 3


def test_longest_still_wins_among_attention_bearing_lists():
    """The length rule is only overruled by the attention filter, not dropped."""

    class TwoStacks(nn.Module):
        def __init__(self):
            super().__init__()
            self.short = nn.ModuleList([_Layer(2) for _ in range(2)])
            self.long = nn.ModuleList([_Layer(2) for _ in range(5)])

    name, blocks = discover_blocks(TwoStacks())
    assert name == "long" and len(blocks) == 5


def test_a_model_without_attention_warns_before_guessing():
    class NoAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.things = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    with pytest.warns(RuntimeWarning, match="expert list"):
        name, _ = discover_blocks(NoAttention())
    assert name == "things"


# ------------------------------------------------------------- forward kwargs


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(2, 2))


def _adapter(block_length: int = 4) -> LLaDA2MoEAdapter:
    cfg = DLLMQuantConfig(
        model_type="llada2_moe",
        tmas=TMASConfig(gen_length=8, block_length=block_length, steps=4),
    )
    adapter = LLaDA2MoEAdapter.__new__(LLaDA2MoEAdapter)  # no checkpoint needed
    adapter.cfg = cfg
    adapter.mask_id = 0
    adapter.block_length = block_length
    adapter.model = _Tiny()
    return adapter


def test_llada15_forward_is_left_exactly_as_it_was():
    """The default must stay empty: model(x) is what produced the measured
    84.5 / 83.0 / 78.0, and an extra kwarg would change that call."""

    class Plain(ModelAdapter):
        def load(self):
            pass

        @property
        def blocks(self):
            return nn.ModuleList()

        def make_probe(self, block):
            raise NotImplementedError

    assert Plain().forward_kwargs(torch.zeros(1, 8, dtype=torch.long)) == {}


def test_mask_is_additive_and_block_lower_triangular():
    adapter = _adapter(block_length=4)
    mask = adapter.forward_kwargs(torch.zeros(1, 8, dtype=torch.long))["attention_mask"]

    assert mask.shape == (1, 1, 8, 8)
    m = mask[0, 0]
    # Within a block: bidirectional, both directions open.
    assert m[0, 3] == 0 and m[3, 0] == 0
    # A later block sees an earlier one ...
    assert m[4, 0] == 0
    # ... and never the reverse.
    assert m[0, 4] == -math.inf
    # Additive, so the only values are 0 and -inf.
    assert set(m.unique().tolist()) == {0.0, -math.inf}


def test_mask_matches_batch_size_because_the_model_compares_shapes():
    """LLaDA2.0 tests the mask shape for equality, so broadcasting is refused."""
    adapter = _adapter(block_length=4)
    ids = torch.zeros(3, 8, dtype=torch.long)
    mask = adapter.forward_kwargs(ids)["attention_mask"]
    assert mask.shape == (3, 1, 8, 8)


def test_no_row_is_entirely_masked():
    """An all -inf row makes softmax return NaN rather than an error."""
    adapter = _adapter(block_length=4)
    m = adapter.forward_kwargs(torch.zeros(1, 10, dtype=torch.long))["attention_mask"]
    assert torch.isfinite(m[0, 0]).any(dim=-1).all()


# -------------------------------------------------------------- block layout


def test_llada15_layout_still_starts_at_the_prompt():
    class Plain(ModelAdapter):
        def load(self):
            pass

        @property
        def blocks(self):
            return nn.ModuleList()

        def make_probe(self, block):
            raise NotImplementedError

    cfg = TMASConfig(gen_length=8, block_length=4, steps=4)
    total, bounds = Plain()._sequence_layout(10, cfg)
    assert total == 18
    assert bounds == [(10, 14), (14, 18)]


def test_llada2_blocks_sit_on_a_grid_anchored_at_zero():
    """A prompt that ends mid-block is decoded together with that block.

    Anchoring at the prompt instead would put a decoding block across a grid
    boundary, and its first half would lose sight of its second -- not a
    crash, just quietly worse text.
    """
    adapter = _adapter(block_length=4)
    cfg = TMASConfig(gen_length=8, block_length=4, steps=4)
    total, bounds = adapter._sequence_layout(10, cfg)

    assert total == 20  # rounded up to whole blocks
    assert bounds == [(8, 12), (12, 16), (16, 20)]
    assert all(lo % 4 == 0 for lo, _ in bounds)


def test_layout_updates_the_mask_grid():
    """The sampler and the mask have to agree on one block length."""
    adapter = _adapter(block_length=4)
    adapter._sequence_layout(10, TMASConfig(gen_length=8, block_length=8, steps=4))
    assert adapter.block_length == 8


# --------------------------------------------------------- expert coverage


class _Gate(nn.Module):
    """LLaDA2.0's router: a bare Parameter, and it returns its own top-k."""

    def __init__(self, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight = nn.Parameter(torch.zeros(num_experts, 4))
        self.routed = torch.tensor([[0, 1], [0, 2]])

    def forward(self, hidden_states):
        logits = torch.zeros(2, self.num_experts)
        return self.routed, torch.ones_like(self.routed, dtype=torch.float), logits


class _MoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _Gate()

    def forward(self, x):
        return self.gate(x)


def test_a_router_that_is_not_a_linear_is_still_found():
    """The isinstance(nn.Linear) check found zero routers in 19 MoE layers."""
    from dllmquant.models.llada2_moe import ExpertCoverage

    model = _MoEBlock()
    cov = ExpertCoverage().attach(model)
    model(torch.zeros(1, 2, 4))
    cov.detach()

    assert "gate" in cov.counts


def test_coverage_counts_every_route_not_just_the_strongest():
    """Each of a token's k routes puts it in that expert's Hessian."""
    from dllmquant.models.llada2_moe import ExpertCoverage

    model = _MoEBlock()
    cov = ExpertCoverage().attach(model)
    model(torch.zeros(1, 2, 4))
    cov.detach()

    # routed = [[0, 1], [0, 2]] -> expert 0 twice, 1 and 2 once, 3 starved.
    assert cov.counts["gate"].tolist() == [2, 1, 1, 0]
    assert cov.exact["gate"] is True
    assert "top-k routes" in cov.report(min_tokens=1)


def test_a_linear_router_falls_back_to_top1_and_says_so():
    from dllmquant.models.llada2_moe import ExpertCoverage

    class LinearRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(4, 3, bias=False)

        def forward(self, x):
            return self.gate(x)

    model = LinearRouter()
    with torch.no_grad():
        model.gate.weight.copy_(torch.tensor(
            [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]]))

    cov = ExpertCoverage().attach(model)
    model(torch.eye(4)[:2])  # picks expert 0, then 1
    cov.detach()

    assert cov.counts["gate"].tolist() == [1, 1, 0]
    assert cov.exact["gate"] is False
    assert "top-1 estimate" in cov.report(min_tokens=1)


# ------------------------------------------------------------ sampler runs


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _StrictModel(nn.Module):
    """Stands in for LLaDA2.0 by being just as unforgiving about the mask.

    The real remote code reads ``attention_mask.size()`` with no None check and
    compares the shape for equality, which is how a missing mask became an
    AttributeError twelve frames down instead of a clear message.
    """

    def __init__(self, vocab: int = 16, d: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is None:
            raise AttributeError("'NoneType' object has no attribute 'size'")
        b, t = input_ids.shape
        if tuple(attention_mask.size()) != (b, 1, t, t):
            raise ValueError(
                f"LLaDA2.0 only supports a block mask of {(b, 1, t, t)}, "
                f"got {tuple(attention_mask.size())}"
            )
        return _Out(self.head(self.embed(input_ids)))


def test_the_sampler_runs_end_to_end_against_the_strict_contract():
    adapter = _adapter(block_length=4)
    adapter.model = _StrictModel()
    adapter.mask_id = 15

    cfg = TMASConfig(gen_length=8, block_length=4, steps=6)
    prompt = torch.tensor([1, 2, 3, 4, 5, 6])  # 6 tokens: ends mid-block
    out = adapter.generate(prompt, cfg)

    assert out.shape[-1] == 16  # 6 + 8 rounded up to whole blocks
    assert (out[0, prompt.shape[-1]:] != adapter.mask_id).all()


def test_snapshots_cover_every_step_when_blocks_do_not_divide_steps():
    """The block count follows from the prompt length, which no caller picks."""
    adapter = _adapter(block_length=4)
    adapter.model = _StrictModel()
    adapter.mask_id = 15

    cfg = TMASConfig(gen_length=8, block_length=4, steps=7)
    snaps = adapter.trace_generation(torch.tensor([1, 2, 3, 4, 5, 6]), cfg)

    assert len(snaps) == 7
    assert [s.step for s in snaps] == list(range(7))


# ---------------------------------------------------------- attention probe


class _Block(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.query_key_value = nn.Linear(d, 3 * d, bias=False)
        self.attn_norm = nn.LayerNorm(d)


def _probe(head_dim=8, rotary_dim=None, attn_mask_fn=None):
    from dllmquant.models.llada import LLaDAAttentionProbe

    return LLaDAAttentionProbe(
        _Block(head_dim), n_heads=1, n_kv_heads=1, head_dim=head_dim,
        rotary=None, rope_theta=10000.0,
        rotary_dim=rotary_dim, attn_mask_fn=attn_mask_fn,
    )


def test_partial_rotary_leaves_the_unrotated_half_alone():
    """LLaDA2.0 rotates 64 of 128 channels; rotating all 128 changes Q/K."""
    probe = _probe(head_dim=8, rotary_dim=4)
    q = torch.randn(1, 1, 3, 8)
    q_out, _ = probe._apply_rotary(q, q.clone(), 3)

    assert torch.equal(q_out[..., 4:], q[..., 4:])
    assert not torch.allclose(q_out[..., :4], q[..., :4])
    assert probe.rope_applied


def test_full_rotary_is_untouched_for_llada15():
    probe = _probe(head_dim=8, rotary_dim=None)
    q = torch.randn(1, 1, 3, 8)
    q_out, _ = probe._apply_rotary(q, q.clone(), 3)
    assert not torch.allclose(q_out, q)  # every channel took part


def test_probe_respects_the_block_mask():
    """Without it the rows still sum to 1, so a wrong ranking looks healthy."""
    adapter = _adapter(block_length=4)
    probe = _probe(head_dim=8, attn_mask_fn=adapter._probe_attn_mask_fn())

    q = torch.randn(1, 1, 8, 8)
    probs = probe._attention_probs(q, torch.randn(1, 1, 8, 8))

    # A query in the first block puts no weight on the second.
    assert torch.allclose(probs[0, 0, 0, 4:], torch.zeros(4))
    # A query in the second block sees both.
    assert probs[0, 0, 4, 0] > 0
    assert torch.allclose(probs.sum(-1), torch.ones(1, 1, 8), atol=1e-5)


def test_unmasked_probe_attends_everywhere():
    probe = _probe(head_dim=8)
    probs = probe._attention_probs(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    assert (probs > 0).all()


def test_blocks_fully_inside_the_prompt_are_not_decoded():
    adapter = _adapter(block_length=4)
    _, bounds = adapter._sequence_layout(8, TMASConfig(
        gen_length=4, block_length=4, steps=2))
    assert bounds == [(8, 12)]
