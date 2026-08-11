"""Adapter for the dense LLaDA family (LLaDA-8B-*, LLaDA-1.5-8B).

LLaDA is an OLMo-style decoder with **bidirectional** attention -- there is no
causal mask, which is precisely why the value matrix is shared across the whole
sequence and why IA-AQ has something to optimise.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from ..config import DLLMQuantConfig
from .base import (
    DOWN_PROJECTION_NAMES,
    ArchitectureMismatch,
    AttentionParts,
    AttentionProbe,
    ModelAdapter,
    discover_blocks,
    find_submodule,
    load_pretrained,
)

# LLaDA's reserved mask token; identical for LLaDA-8B and LLaDA-1.5.
LLADA_MASK_ID = 126336

# `query_key_value` / `dense` are LLaDA2.0's names (Bailing-MoE lineage);
# `att_proj` / `attn_out` are LLaDA-1.5's (OLMo lineage). The two families
# share no attention naming at all.
_QKV_FUSED_NAMES = ("att_proj", "qkv_proj", "Wqkv", "query_key_value")
_Q_NAMES = ("q_proj", "wq", "query")
_K_NAMES = ("k_proj", "wk", "key")
_V_NAMES = ("v_proj", "wv", "value")
# The norm applied to the block input *before* the QKV projection.  Listed
# explicitly rather than matching a bare "norm", which would also catch the
# MLP's norm and silently feed the projection the wrong tensor.
_ATTN_NORM_NAMES = ("attn_norm", "input_layernorm", "ln_1", "norm1", "ln_attn")
_FF_NORM_NAMES = ("ff_norm", "post_attention_layernorm", "ln_2", "norm2", "ln_mlp")
# Ordered by specificity: the first name present wins, so a model that has
# both `ln_f` and something generically called `norm` outside its blocks keeps
# the one it meant.  LLaDA2.0's is the bare `norm`, which is why it is here at
# all -- and why it is last.
_FINAL_NORM_NAMES = ("ln_f", "final_layernorm", "norm_f", "final_norm", "norm")
_EMBED_NAMES = ("wte", "embed_tokens", "embed_in", "word_embeddings")
_LM_HEAD_NAMES = ("ff_out", "lm_head", "embed_out")

# Residual-stream classification for QuaRot. By name, deliberately: `attn_out`
# is square, so no shape rule can separate it from `att_proj`.
_ATTN_IN_NAMES = _QKV_FUSED_NAMES + _Q_NAMES + _K_NAMES + _V_NAMES
_RESIDUAL_IN_NAMES = _ATTN_IN_NAMES + (
    "ff_proj", "gate_proj", "up_proj", "w1", "w3",
)
_ATTN_OUT_NAMES = ("attn_out", "o_proj", "out_proj", "dense")
_RESIDUAL_OUT_NAMES = _ATTN_OUT_NAMES + DOWN_PROJECTION_NAMES

# Public, and derived rather than retyped. Three places used to keep their own
# copy of these lists and the copies drifted: selfcheck reported "IA-AQ would
# be a no-op" for a model whose fused projection is simply called
# `query_key_value`, and the pipeline -- holding a third copy -- then really
# did make it one, silently, for a whole 20-block run. Anything that needs to
# recognise a projection by name reads it from here.
QKV_FUSED_NAMES = _QKV_FUSED_NAMES
V_PROJECTION_NAMES = _V_NAMES
VALUE_BEARING_NAMES = QKV_FUSED_NAMES + V_PROJECTION_NAMES

ATTENTION_IN_NAMES = _ATTN_IN_NAMES
ATTENTION_OUT_NAMES = _ATTN_OUT_NAMES
MLP_IN_NAMES = tuple(n for n in _RESIDUAL_IN_NAMES if n not in _ATTN_IN_NAMES)
MLP_OUT_NAMES = tuple(n for n in _RESIDUAL_OUT_NAMES if n not in _ATTN_OUT_NAMES)


class LLaDAAttentionProbe(AttentionProbe):
    """Recovers Q/K/V from a block's projections and forms the softmax matrix.

    The attention probabilities are recomputed here instead of being requested
    via ``output_attentions=True``: the remote-code LLaDA modelling file routes
    through fused SDPA/flash kernels that never materialise the matrix.

    Caveat, stated plainly: rotary embeddings are applied only when the block
    exposes a rotary module we recognise.  Without it the recovered
    probabilities ignore positional rotation, which shifts the *ranking* of
    token importance slightly.  IA-AQ uses these weights as a soft priority,
    not as an exact quantity, but the fallback is reported by ``selfcheck``.
    """

    def __init__(
        self,
        block: nn.Module,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rotary: Optional[nn.Module] = None,
        rope_theta: float = 0.0,
        rotary_dim: Optional[int] = None,
        attn_mask_fn=None,
    ):
        super().__init__(block)
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rotary = rotary
        self.rope_theta = rope_theta
        self.rope_applied = False
        # LLaDA2.0 rotates only the first half of each head (partial_rotary_
        # factor 0.5) and leaves the rest untouched; LLaDA-1.5 rotates all of
        # it.  Rotating channels the model never rotates changes Q/K and so
        # the importance ranking IA-AQ reads off the softmax.
        self.rotary_dim = rotary_dim or head_dim
        # Block-diffusion attention is not bidirectional across blocks.  With
        # no mask the probe would credit tokens the model cannot see.
        self.attn_mask_fn = attn_mask_fn

        self.fused = find_submodule(block, _QKV_FUSED_NAMES)
        self.q = find_submodule(block, _Q_NAMES)
        self.k = find_submodule(block, _K_NAMES)
        self.v = find_submodule(block, _V_NAMES)
        self.attn_norm = find_submodule(block, _ATTN_NORM_NAMES)
        if self.attn_norm is None:
            warnings.warn(
                f"no pre-attention norm found among {_ATTN_NORM_NAMES}; the probe "
                "will project the raw block input, which is wrong for any "
                "pre-norm transformer. Check the block's children.",
                RuntimeWarning,
            )

        if self.fused is None and None in (self.q, self.k, self.v):
            raise ArchitectureMismatch(
                "block exposes neither a fused QKV projection "
                f"{_QKV_FUSED_NAMES} nor separate {_Q_NAMES}/{_K_NAMES}/{_V_NAMES}; "
                f"children: {[n for n, _ in block.named_children()]}"
            )

    # ------------------------------------------------------------------ hook

    def _hook(self, module, args, kwargs, output) -> None:
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if hidden is None:
            raise ArchitectureMismatch("could not locate block hidden_states input")
        if hidden.dim() != 3:
            return  # not the call we care about

        with torch.no_grad():
            q, k, v = self._project(hidden)
            probs = self._attention_probs(q, k)
        self.parts = AttentionParts(value_states=v, attn_probs=probs)

    def _project(self, hidden: torch.Tensor):
        b, t, _ = hidden.shape
        # Pre-norm transformer: the projection sees the normed input, not the
        # residual stream. Skipping this silently rescales Q/K and distorts the
        # softmax the importance weights are read from.
        if self.attn_norm is not None:
            hidden = self.attn_norm(hidden)
        if self.fused is not None:
            qkv = self.fused(hidden)
            d_q = self.n_heads * self.head_dim
            d_kv = self.n_kv_heads * self.head_dim
            if qkv.shape[-1] != d_q + 2 * d_kv:
                raise ArchitectureMismatch(
                    f"fused QKV width {qkv.shape[-1]} != "
                    f"{d_q} + 2*{d_kv}; head config is wrong"
                )
            q, k, v = qkv.split([d_q, d_kv, d_kv], dim=-1)
        else:
            q, k, v = self.q(hidden), self.k(hidden), self.v(hidden)

        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.rotary is not None:
            q, k = self._apply_rotary(q, k, t)
        return q, k, v

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _rope_tables(self, seq_len: int, device, dtype):
        """Llama-convention RoPE, built from the config's theta.

        LLaDA applies the rotation inside its attention function rather than
        through a submodule, so there is nothing to borrow -- but the tables
        are three lines, and computing them is far better than dropping the
        positional rotation and calling the resulting attention weights
        "approximate".
        """
        dim = self.rotary_dim
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _apply_rotary(self, q, k, seq_len):
        cos = sin = None
        if self.rotary is not None:
            for call in (lambda: self.rotary(q, seq_len=seq_len),
                         lambda: self.rotary(seq_len)):
                try:
                    cos, sin = call()
                    break
                except Exception:
                    continue
        if cos is None:
            if not self.rope_theta:
                return q, k
            cos, sin = self._rope_tables(seq_len, q.device, q.dtype)

        cos = cos.to(q.dtype).reshape(1, 1, seq_len, -1)
        sin = sin.to(q.dtype).reshape(1, 1, seq_len, -1)
        self.rope_applied = True

        # Partial rotary: the width of the tables says how much of the head is
        # rotated; the remainder passes through unchanged.  When they cover the
        # whole head -- LLaDA-1.5 -- the split is a no-op.
        rd = min(cos.shape[-1], q.shape[-1])
        if rd == q.shape[-1]:
            return (
                q * cos + self._rotate_half(q) * sin,
                k * cos + self._rotate_half(k) * sin,
            )

        def rotate(x):
            head, tail = x[..., :rd], x[..., rd:]
            return torch.cat(
                (head * cos + self._rotate_half(head) * sin, tail), dim=-1
            )

        return rotate(q), rotate(k)

    def _attention_probs(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Grouped-query attention: repeat KV heads to match query heads.
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
        scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
        # LLaDA-1.5 attends bidirectionally over the whole sequence, so there is
        # nothing to mask.  Block-diffusion models supply a mask here: without
        # it the probe would hand IA-AQ weights for tokens the model is blind
        # to, and the rows would still sum to 1, so nothing would look wrong.
        if self.attn_mask_fn is not None:
            mask = self.attn_mask_fn(q.shape[-2], q.device, torch.float32)
            if mask is not None:
                scores = scores + mask
        return torch.softmax(scores, dim=-1)


class LLaDAAdapter(ModelAdapter):
    """LLaDA-8B-Base / -Instruct / LLaDA-1.5-8B."""

    # Remote code written against transformers 4.38.2; breaks from 4.47 on.
    TRANSFORMERS_MIN = (4, 38)
    TRANSFORMERS_MAX = (4, 47)

    def __init__(self, cfg: DLLMQuantConfig):
        self.cfg = cfg
        self.mask_id = LLADA_MASK_ID
        self.model = None
        self.tokenizer = None
        self._blocks = None
        self._blocks_path = ""
        self.n_heads = 0
        self.n_kv_heads = 0
        self.head_dim = 0

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_path, trust_remote_code=True
        )
        self.model = load_pretrained(
            AutoModel, self.cfg, self.TRANSFORMERS_MIN, self.TRANSFORMERS_MAX
        )
        self.model.eval()
        self._validate()

    def _validate(self) -> None:
        self._blocks_path, self._blocks = discover_blocks(self.model)

        cfg = self.model.config
        self.n_heads = self._cfg_get(cfg, ["n_heads", "num_attention_heads"])
        hidden = self._cfg_get(cfg, ["d_model", "hidden_size"])
        self.n_kv_heads = self._cfg_get(
            cfg, ["n_kv_heads", "num_key_value_heads"], default=self.n_heads
        )

        # Read, not derived. `hidden // n_heads` happens to be right for both
        # LLaDA families, but a config that states head_dim outright is stating
        # it for a reason, and a model where the two differ would get every
        # attention-shaped reshape wrong without raising anything.
        declared = getattr(cfg, "head_dim", None)
        self.head_dim = int(declared) if declared else hidden // self.n_heads

        # The residual stream is `hidden` wide. That is the same as
        # n_heads * head_dim in both families here, and code elsewhere has
        # taken the product as a stand-in; keep the real number so it does not
        # have to.
        self.d_model = int(hidden)

        if self.n_heads % self.n_kv_heads != 0:
            raise ArchitectureMismatch(
                f"n_heads={self.n_heads} not divisible by n_kv_heads={self.n_kv_heads}"
            )

        vocab = self._cfg_get(cfg, ["vocab_size", "embedding_size"], default=0)
        if vocab and self.mask_id >= vocab:
            raise ArchitectureMismatch(
                f"mask_id {self.mask_id} outside vocab of {vocab}; "
                "this does not look like a LLaDA checkpoint"
            )

    @staticmethod
    def _cfg_get(cfg, names: Sequence[str], default=None):
        for n in names:
            if hasattr(cfg, n):
                return getattr(cfg, n)
        if default is not None:
            return default
        raise ArchitectureMismatch(
            f"model config exposes none of {names}; available: "
            f"{sorted(k for k in vars(cfg) if not k.startswith('_'))[:20]}"
        )

    # ------------------------------------------------------------- structure

    @property
    def blocks(self) -> nn.ModuleList:
        if self._blocks is None:
            raise RuntimeError("call load() first")
        return self._blocks

    def _probe_rotary_dim(self) -> Optional[int]:
        """How much of each head carries rotary. Full head for LLaDA-1.5."""
        return None

    def _probe_attn_mask_fn(self):
        """Additive attention mask for the probe, or None for full attention."""
        return None

    def make_probe(self, block: nn.Module) -> LLaDAAttentionProbe:
        rotary = find_submodule(block, ("rotary_emb", "rope", "rotary"))
        if rotary is None:
            rotary = find_submodule(self.model, ("rotary_emb", "rope", "rotary"))
        theta = float(
            getattr(self.model.config, "rope_theta", 0.0)
            or getattr(self.model.config, "rope_base", 0.0)
            or 0.0
        )
        probe = LLaDAAttentionProbe(
            block, self.n_heads, self.n_kv_heads, self.head_dim, rotary, theta,
            rotary_dim=self._probe_rotary_dim(),
            attn_mask_fn=self._probe_attn_mask_fn(),
        )
        if rotary is None and not theta:
            warnings.warn(
                "no rotary module and no rope_theta in the config; IA-AQ "
                "importance weights will ignore positional rotation "
                "(approximate ranking)",
                RuntimeWarning,
            )
        return probe

    # -------------------------------------------------------------- rotation

    def rotation_plan(self):
        """Classify every linear as reading or writing the residual stream.

        Classification is by name, not by shape: ``attn_out`` has
        ``in_features == out_features == d_model``, so a shape rule cannot tell
        it apart from ``att_proj`` -- and getting that backwards silently
        breaks invariance instead of raising.
        """
        from ..algos.quarot import RotationPlan
        from ..modules import QuantLinear

        def is_linear(m) -> bool:
            # QuantLinear counts too: rotating a model whose linears are already
            # wrapped is legitimate, and it has the same weight/in_features/
            # out_features surface the rotation needs.
            return isinstance(m, (nn.Linear, QuantLinear))

        blocks_prefix = self._blocks_path + "."
        embeddings = [
            m for name, m in self.model.named_modules()
            if isinstance(m, nn.Embedding)
            and name.split(".")[-1] in _EMBED_NAMES
        ]
        if not embeddings:
            raise ArchitectureMismatch(
                f"no token embedding found among names {_EMBED_NAMES}"
            )

        input_linears: List[nn.Linear] = []
        output_linears: List[nn.Linear] = []
        norm_groups = []
        head_pairs = []

        for block in self.blocks:
            ins, outs = [], []
            for name, module in block.named_modules():
                leaf = name.split(".")[-1]
                if not is_linear(module):
                    continue
                if leaf in _RESIDUAL_IN_NAMES:
                    ins.append((leaf, module))
                elif leaf in _RESIDUAL_OUT_NAMES:
                    outs.append((leaf, module))

            # Modules that read the residual stream without being nn.Linear.
            # Empty for LLaDA-1.5; an MoE router lives here.
            ins.extend(self._extra_residual_readers(block))

            if not ins or not outs:
                raise ArchitectureMismatch(
                    f"block exposes {[n for n, _ in ins]} inputs and "
                    f"{[n for n, _ in outs]} outputs; expected at least one of each"
                )

            # Structural coverage is the real guard against a missed layer.
            # The numeric invariance check cannot serve that role in bf16: the
            # rotation itself perturbs the output by a few percent there, so a
            # tolerance loose enough to pass would also hide a genuine miss.
            classified = {id(m) for _, m in ins} | {id(m) for _, m in outs}
            unclassified = [
                name for name, module in block.named_modules()
                if is_linear(module) and id(module) not in classified
            ]
            if unclassified:
                raise ArchitectureMismatch(
                    f"these linears in the block are neither residual readers "
                    f"nor writers, so rotation would break the model: "
                    f"{unclassified}. Add their leaf names to "
                    f"_RESIDUAL_IN_NAMES or _RESIDUAL_OUT_NAMES in "
                    f"dllmquant/models/llada.py"
                )
            input_linears.extend(m for _, m in ins)
            output_linears.extend(m for _, m in outs)

            attn_norm = find_submodule(block, _ATTN_NORM_NAMES)
            ff_norm = find_submodule(block, _FF_NORM_NAMES)
            attn_consumers = [m for n, m in ins if n in _ATTN_IN_NAMES]
            ff_consumers = [m for n, m in ins if n not in _ATTN_IN_NAMES]
            if attn_norm is not None and attn_consumers:
                norm_groups.append((attn_norm, attn_consumers))
            if ff_norm is not None and ff_consumers:
                norm_groups.append((ff_norm, ff_consumers))

            out_proj = next((m for n, m in outs if n in _ATTN_OUT_NAMES), None)
            fused = next((m for n, m in ins if n in _QKV_FUSED_NAMES), None)
            v_only = next((m for n, m in ins if n in _V_NAMES), None)
            if out_proj is not None and fused is not None:
                offset = (self.n_heads + self.n_kv_heads) * self.head_dim
                head_pairs.append((fused, out_proj, offset))
            elif out_proj is not None and v_only is not None:
                head_pairs.append((v_only, out_proj, None))

        # The LM head reads the residual stream, so it rotates like an input.
        # In LLaDA it is called `ff_out` -- the same leaf name as the MLP
        # down-projection inside every block -- so it must be located strictly
        # outside the block list.
        head = None
        norm_candidates = {}
        outside_norms = set()
        for name, module in self.model.named_modules():
            if name.startswith(blocks_prefix) or name == self._blocks_path:
                continue
            leaf = name.split(".")[-1]
            if is_linear(module) and leaf in _LM_HEAD_NAMES:
                head = module
                continue
            if not is_linear(module) and hasattr(module, "weight"):
                if getattr(module, "weight", None) is not None:
                    outside_norms.add(leaf)
                if leaf in _FINAL_NORM_NAMES:
                    norm_candidates.setdefault(leaf, module)

        final_norm = next(
            (norm_candidates[n] for n in _FINAL_NORM_NAMES if n in norm_candidates),
            None,
        )

        if head is not None:
            input_linears.append(head)
            if final_norm is None:
                # Skipping quietly is what turns a naming mismatch into a
                # wrong number: the head rotates, the norm's per-channel
                # weight does not follow, and the only symptom is an
                # invariance error large enough to look like a deeper bug.
                raise ArchitectureMismatch(
                    "the LM head rotates with the residual stream, but no "
                    f"final norm was found among {_FINAL_NORM_NAMES} outside "
                    "the blocks. Its weight has to be fused into the head "
                    "before rotating, or invariance breaks. Norm-like modules "
                    f"outside the blocks: {sorted(outside_norms)}"
                )
            norm_groups.append((final_norm, [head]))

        return RotationPlan(
            embeddings=embeddings,
            input_linears=input_linears,
            output_linears=output_linears,
            norm_groups=norm_groups,
            head_pairs=head_pairs,
        )

    def describe(self) -> str:
        return (
            f"LLaDAAdapter: {len(self.blocks)} blocks at '{self._blocks_path}', "
            f"heads={self.n_heads}, kv_heads={self.n_kv_heads}, "
            f"head_dim={self.head_dim}, mask_id={self.mask_id}"
        )


__all__ = ["LLaDAAdapter", "LLaDAAttentionProbe", "LLADA_MASK_ID"]
