"""Adapter for LLaDA2.0 (inclusionAI / Ant Group) -- MoE diffusion LMs.

LLaDA2.0-mini is 16B-A1B and -flash is 100B-A6B, so only ~6% of the parameters
see any given token.  That breaks an assumption CGQ inherits from GPTQ: the
Hessian of an expert's projection is estimated from the tokens routed to *that*
expert, not from the whole calibration set.  With 128 snapshots x 256 tokens
and 64 experts picking top-2, an expert sees roughly

    128 * 256 * 2 / 64  ~  1000 tokens

against an input dimension in the thousands -- a rank-deficient Hessian.  This
adapter therefore reports per-expert token coverage so the calibration budget
can be raised before a long run, and the solver's damping backoff keeps the
factorization alive when coverage is thin.

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
from typing import Dict, List, Optional

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
    """Counts how many calibration tokens reach each expert."""

    def __init__(self):
        self.counts: Dict[str, torch.Tensor] = {}
        self._handles: List = []

    def attach(self, model: nn.Module) -> "ExpertCoverage":
        for name, module in model.named_modules():
            if name.split(".")[-1] in _ROUTER_NAMES and isinstance(module, nn.Linear):
                self._handles.append(
                    module.register_forward_hook(self._make_hook(name))
                )
        return self

    def _make_hook(self, name: str):
        def hook(module, args, output):
            logits = output if isinstance(output, torch.Tensor) else output[0]
            if logits.dim() < 2:
                return
            n_experts = logits.shape[-1]
            flat = logits.reshape(-1, n_experts)
            # Top-1 is enough to spot starvation; top-k only scales it.
            top = flat.argmax(dim=-1)
            counts = torch.bincount(top.cpu(), minlength=n_experts)
            if name in self.counts:
                self.counts[name] = self.counts[name] + counts
            else:
                self.counts[name] = counts

        return hook

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def report(self, min_tokens: int = 512) -> str:
        if not self.counts:
            return "no routers observed -- is this actually an MoE checkpoint?"
        lines = ["expert coverage (top-1 routing):"]
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
            lines.append(
                f"  !! {starved_total} expert projections will be solved from an "
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
