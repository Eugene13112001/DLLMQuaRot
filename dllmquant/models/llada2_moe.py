"""Adapter for LLaDA2.0 (inclusionAI / Ant Group) -- MoE diffusion LMs.

LLaDA2.0-mini is 16B-A1B and -flash is 100B-A6B, so only ~6% of the parameters
see any given token.  That breaks an assumption CGQ inherits from GPTQ: the
Hessian of an expert's projection is estimated from the tokens routed to *that*
expert, not from the whole calibration set.  mini's own config -- 256 experts,
top-8, hidden 2048 -- puts a number on it.  With 128 snapshots of 256 tokens:

    128 * 256 * 8 / 256  ~  1000 tokens per expert

``gate_proj`` and ``up_proj`` take the full hidden width, so their Hessian is
2048x2048 and a thousand tokens leaves it rank-deficient by a factor of two.
``down_proj`` reads the 512-wide expert intermediate and is comfortable at the
same budget.  So the shortfall is specific, not general, and it is the reason
this adapter reports per-expert coverage before a long run rather than after;
the solver's damping backoff keeps the factorization alive when it is thin.

Two further caveats, both surfaced by ``scripts/selfcheck.py`` rather than
silently absorbed:

* The mask token id is read from the checkpoint, not hardcoded -- LLaDA2.0 does
  not share LLaDA-1.5's tokenizer.
* Certainty weights are matched to activations by token count.  Expert
  projections receive a gathered subset of the sequence, so the count will not
  match and those layers fall back to uniform weights.  Routing-aware weighting
  needs the router's index tensor; ``ExpertCoverage`` collects it, but wiring it
  into the per-expert Hessian is left explicit rather than guessed.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import DLLMQuantConfig
from .base import (
    ArchitectureMismatch,
    discover_blocks,
    find_submodule,
    load_pretrained,
)
from .llada import LLaDAAdapter

_ROUTER_NAMES = ("gate", "router", "wg", "gate_proj_moe")
_EXPERT_CONTAINER_NAMES = ("experts", "mlp_experts", "moe_experts")


class ExpertCoverage:
    """Counts how many calibration tokens reach each expert.

    A router is not necessarily an ``nn.Linear``: LLaDA2.0's ``gate`` holds a
    bare ``nn.Parameter`` and calls ``F.linear`` itself, which is why an
    isinstance check on Linear found no routers at all on a checkpoint with
    nineteen of them.  It is recognised by what it exposes instead.
    """

    def __init__(self):
        self.counts: Dict[str, torch.Tensor] = {}
        self.exact: Dict[str, bool] = {}
        self._handles: List = []

    @staticmethod
    def _is_router(module: nn.Module) -> bool:
        if isinstance(module, nn.Linear):
            return True
        # A routing module announces the shape of its decision.
        return hasattr(module, "num_experts") and hasattr(module, "top_k")

    def attach(self, model: nn.Module) -> "ExpertCoverage":
        for name, module in model.named_modules():
            if name.split(".")[-1] in _ROUTER_NAMES and self._is_router(module):
                self._handles.append(
                    module.register_forward_hook(self._make_hook(name))
                )
        return self

    @staticmethod
    def _routed_experts(output) -> Tuple[Optional[torch.Tensor], bool, int]:
        """Which experts got tokens, whether that is exact, and how many exist.

        An integer tensor is the router's own top-k choice, counted as is:
        every route, not just the strongest, because every route puts a token
        into that expert's Hessian.  Float logits are all a plain Linear
        router gives, and there top-1 is an estimate, not the answer.

        The expert count is returned separately because it cannot be recovered
        from the routes: an expert that received nothing is exactly the one
        worth reporting, and inferring the count from ``idx.max()`` would drop
        it off the end of the histogram.
        """
        parts = output if isinstance(output, (tuple, list)) else (output,)
        tensors = [t for t in parts if isinstance(t, torch.Tensor) and t.numel()]
        if not tensors:
            return None, False, 0

        for t in tensors:
            if not t.is_floating_point():
                return t.reshape(-1), True, 0

        logits = tensors[0]
        if logits.dim() < 2:
            return None, False, 0
        width = logits.shape[-1]
        return logits.reshape(-1, width).argmax(dim=-1), False, width

    def _make_hook(self, name: str):
        def hook(module, args, output):
            idx, exact, width = self._routed_experts(output)
            if idx is None:
                return
            n_experts = (
                int(getattr(module, "num_experts", 0)) or width or int(idx.max()) + 1
            )
            counts = torch.bincount(idx.cpu(), minlength=n_experts)
            self.counts[name] = self.counts.get(name, 0) + counts
            self.exact[name] = exact

        return hook

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def report(self, min_tokens: int = 512) -> str:
        if not self.counts:
            return "no routers observed -- is this actually an MoE checkpoint?"
        how = (
            "top-k routes"
            if self.exact and all(self.exact.values())
            else "top-1 estimate"
        )
        lines = [f"expert coverage ({how}):"]
        starved_total = 0
        for name, counts in sorted(self.counts.items()):
            starved = int((counts < min_tokens).sum())
            starved_total += starved
            lines.append(
                f"  {name}: {len(counts)} experts, "
                f"min {int(counts.min())}, median {int(counts.median())}, "
                f"max {int(counts.max())} tokens; {starved} below {min_tokens}"
            )
        if starved_total:
            total = sum(len(c) for c in self.counts.values())
            lines.append(
                f"  !! {starved_total} of {total} expert projections saw fewer "
                f"than {min_tokens} tokens and would be solved from an "
                f"under-determined Hessian. Raise tmas.n_samples or "
                f"tmas.gen_length, or quantize starved experts with RTN."
            )
        return "\n".join(lines)


class LLaDA2MoEAdapter(LLaDAAdapter):
    """LLaDA2.0-mini (16B-A1B) and LLaDA2.0-flash (100B-A6B)."""

    # `modeling_llada2_moe.py` imports `dynamic_rope_update` (the RoPE refactor)
    # and `TransformersKwargs`; neither exists before 4.56.  No upper bound --
    # unlike LLaDA-1.5's, this remote code is recent enough that no later
    # release has been shown to break it.  Note the two windows do not overlap
    # with LLaDA-1.5's (4.38-4.46), so one venv cannot serve both models.
    TRANSFORMERS_MIN = (4, 56)
    TRANSFORMERS_MAX = None

    def __init__(self, cfg: DLLMQuantConfig):
        super().__init__(cfg)
        self.mask_id = -1  # discovered from the checkpoint
        self.coverage = ExpertCoverage()
        # The attention mask and the sampler must be built on the *same* block
        # grid, so the two share this one number.  ``_sequence_layout`` updates
        # it from whatever TMASConfig the caller passed, because callers do
        # pass a different one than cfg.tmas (selfcheck, for instance).
        self.block_length = cfg.tmas.block_length

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_path, trust_remote_code=True
        )
        self.model = load_pretrained(
            AutoModelForCausalLM, self.cfg,
            self.TRANSFORMERS_MIN, self.TRANSFORMERS_MAX,
        )
        self.model.eval()
        self.mask_id = self._discover_mask_id()
        self._validate()

    def _discover_mask_id(self) -> int:
        cfg = self.model.config
        for attr in ("mask_token_id", "mask_id", "mask_token"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val >= 0:
                return val

        tok = self.tokenizer
        if getattr(tok, "mask_token_id", None) is not None:
            return int(tok.mask_token_id)

        for literal in ("<mask>", "[MASK]", "<|mask|>", "<mdm_mask>"):
            ids = tok.convert_tokens_to_ids(literal)
            if isinstance(ids, int) and ids >= 0:
                return ids

        raise ArchitectureMismatch(
            "could not determine the mask token id for this checkpoint. "
            "Set it explicitly: adapter.mask_id = <id>. Check the model card "
            "or tokenizer.added_tokens_decoder for the [MASK] entry."
        )

    # ----------------------------------------------------------- block masking

    def block_attention_mask(
        self,
        seq_len: int,
        block_length: int,
        *,
        batch_size: int = 1,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        """The block-diffusion mask LLaDA2.0's remote code demands.

        Additive, ``[B, 1, T, T]``, ``0`` where a query may attend and ``-inf``
        where it may not: bidirectional inside a block, visible to every
        earlier block, blind to later ones.  Built exactly as the checkpoint's
        own sampler builds it -- ``tril`` over blocks, expanded to tokens, then
        ``log()`` to turn 1/0 into 0/-inf.

        The shape is checked for equality, not broadcast, by the remote code,
        and a 4-D mask is passed through ``_prepare_4d_causal_attention_mask_
        for_sdpa`` untouched -- no causality is added on top -- so this mask is
        the whole story about what attends to what.
        """
        n_blocks = (seq_len + block_length - 1) // block_length
        allowed = torch.tril(torch.ones(n_blocks, n_blocks, device=device))
        expanded = allowed.repeat_interleave(block_length, dim=0).repeat_interleave(
            block_length, dim=1
        )[:seq_len, :seq_len]
        mask = expanded.log().to(dtype)
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

    def forward_kwargs(self, input_ids: torch.Tensor) -> Dict[str, object]:
        """LLaDA2.0 raises on a missing mask instead of assuming full attention.

        Its ``forward`` reads ``attention_mask.size()`` unguarded, so ``None``
        is an AttributeError several frames deep rather than a default.
        """
        ids = input_ids if input_ids.dim() > 1 else input_ids.unsqueeze(0)
        param = next(self.model.parameters())
        return {
            "attention_mask": self.block_attention_mask(
                ids.shape[-1],
                self.block_length,
                batch_size=ids.shape[0],
                device=param.device,
                dtype=param.dtype,
            )
        }

    # -------------------------------------------------------------- rotation

    def _extra_residual_readers(self, block: nn.Module) -> List[tuple]:
        """The routers, which read the residual stream but are not Linear.

        A router computes ``F.linear(h, self.weight)`` from a bare Parameter,
        so it is invisible to a scan for ``nn.Linear`` -- and it was, in both
        directions at once: its weight never took the rotation, and it never
        received the norm gain that the fusion moved out of the norm and into
        the other consumers.  Either alone breaks invariance; both together
        change which experts a token is routed to, which is not a numerical
        error at all.

        It stays out of the *quantization* set regardless -- `gate` is in
        skip_leaf_names -- and that is a separate decision from rotation:
        rotating a weight is exact, quantizing it is not.
        """
        out = []
        for name, module in block.named_modules():
            leaf = name.split(".")[-1]
            if leaf not in _ROUTER_NAMES:
                continue
            weight = getattr(module, "weight", None)
            if weight is not None and weight.dim() == 2:
                # Linear routers are collected here too: the main scan only
                # takes leaf names it knows, and `gate` is not one of them, so
                # a Linear router would otherwise trip the unclassified-layer
                # guard instead of simply being rotated.
                out.append((leaf, module))
        return out

    def routers(self) -> List[nn.Module]:
        return [m for block in self.blocks
                for _, m in self._extra_residual_readers(block)]

    def _probe_rotary_dim(self) -> Optional[int]:
        """LLaDA2.0 rotates ``head_dim * partial_rotary_factor`` channels only."""
        cfg = self.model.config
        rotary_dim = getattr(cfg, "rotary_dim", None)
        if isinstance(rotary_dim, int) and rotary_dim > 0:
            return rotary_dim
        factor = getattr(cfg, "partial_rotary_factor", 1.0) or 1.0
        return int(self.head_dim * factor)

    def _probe_attn_mask_fn(self):
        def mask_fn(seq_len: int, device, dtype):
            return self.block_attention_mask(
                seq_len, self.block_length, device=device, dtype=dtype
            )

        return mask_fn

    def _sequence_layout(self, p_len: int, cfg) -> tuple:
        """Blocks sit on a grid anchored at position 0, not at the prompt end.

        The mask is built from that grid, so a decoding block that straddled a
        grid boundary would lose bidirectionality over its own first half --
        silently, as slightly worse text.  Following the checkpoint's sampler:
        the prompt's own last block is decoded together with the masks that
        share it, and the sequence is rounded up to a whole number of blocks.
        """
        b = cfg.block_length
        self.block_length = b
        n_blocks = (p_len + cfg.gen_length + b - 1) // b
        first = p_len // b  # blocks lying entirely inside the prompt
        bounds = [(i * b, (i + 1) * b) for i in range(first, n_blocks)]
        return n_blocks * b, bounds

    # ------------------------------------------------------------- structure

    def _validate(self) -> None:
        super()._validate()
        experts = self._find_expert_containers()
        if not experts:
            warnings.warn(
                "no expert containers found; this checkpoint looks dense. "
                "Use --model-type llada for dense LLaDA models.",
                RuntimeWarning,
            )
        self._n_expert_containers = len(experts)

    def _find_expert_containers(self) -> List[str]:
        out = []
        for name, module in self.model.named_modules():
            if (
                name.split(".")[-1] in _EXPERT_CONTAINER_NAMES
                and isinstance(module, (nn.ModuleList, nn.ModuleDict))
            ):
                out.append(name)
        return out

    def describe(self) -> str:
        return (
            f"LLaDA2MoEAdapter: {len(self.blocks)} blocks at '{self._blocks_path}', "
            f"heads={self.n_heads}, kv_heads={self.n_kv_heads}, "
            f"head_dim={self.head_dim}, mask_id={self.mask_id}, "
            f"expert containers={getattr(self, '_n_expert_containers', '?')}"
        )


__all__ = ["LLaDA2MoEAdapter", "ExpertCoverage"]
