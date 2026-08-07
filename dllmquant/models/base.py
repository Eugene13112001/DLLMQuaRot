"""Model adapter protocol.

Everything the quantization pipeline needs to know about a specific DLLM lives
behind this interface: where the transformer blocks are, what the mask token
is, how to roll out a denoising trajectory, and how to observe the attention
matrix for IA-AQ.

The adapters below are written against the published architectures but could
not be executed against real weights while writing them.  They therefore
*verify* their assumptions at load time and raise with a precise message rather
than silently quantizing the wrong tensors -- run ``scripts/selfcheck.py``
before a long job.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..calib.tmas import Snapshot
from ..config import TMASConfig


class ArchitectureMismatch(RuntimeError):
    """Raised when the loaded checkpoint does not look like what we expect."""


@dataclass
class AttentionParts:
    """What IA-AQ needs from one attention layer for one forward pass."""

    value_states: torch.Tensor  # [B, heads, K, head_dim] or [B, K, D]
    attn_probs: torch.Tensor  # [B, heads, Q, K], rows sum to 1


def discover_blocks(model: nn.Module) -> Tuple[str, nn.ModuleList]:
    """Find the transformer block list without hardcoding a module path.

    Returns the qualified name and the ModuleList.  Picks the longest
    ``nn.ModuleList`` whose entries all share one class -- for every decoder
    stack that is the layer list and nothing else.
    """
    best: Optional[Tuple[str, nn.ModuleList]] = None
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) < 2:
            continue
        classes = {type(m) for m in module}
        if len(classes) != 1:
            continue
        if best is None or len(module) > len(best[1]):
            best = (name, module)

    if best is None:
        raise ArchitectureMismatch(
            "could not find a transformer block list in this model; "
            "pass the module path explicitly"
        )
    return best


def find_submodule(root: nn.Module, names: Sequence[str]) -> Optional[nn.Module]:
    """Return the first direct-or-nested child whose attribute name matches."""
    for name in names:
        for mod_name, module in root.named_modules():
            if mod_name.split(".")[-1] == name:
                return module
    return None


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """How many tokens to commit at each step, spread as evenly as possible.

    Mirrors LLaDA's reference implementation: the masked budget of a block is
    split over ``steps``, with the remainder front-loaded.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = torch.zeros(mask_num.size(0), steps, device=mask_index.device,
                      dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        out[i, : int(remainder[i])] += 1
    return out


class ModelAdapter(ABC):
    """Bridges one DLLM family to the quantization pipeline."""

    mask_id: int
    model: nn.Module
    tokenizer: object

    # ------------------------------------------------------------ structure

    @abstractmethod
    def load(self) -> None:
        """Instantiate model + tokenizer and validate the architecture."""

    @property
    @abstractmethod
    def blocks(self) -> nn.ModuleList:
        """The transformer blocks, in execution order."""

    @abstractmethod
    def make_probe(self, block: nn.Module) -> "AttentionProbe":
        """A probe that captures V and the softmax matrix for this block.

        Block inputs are captured by the pipeline with a catcher hook rather
        than by re-running the embedding stack, so adapters do not need to know
        what happens before block 0.
        """

    # ------------------------------------------------------------ generation

    @torch.no_grad()
    def trace_generation(
        self, prompt_ids: torch.Tensor, cfg: TMASConfig
    ) -> List[Snapshot]:
        """Roll out a trajectory and return every intermediate state."""
        snapshots: List[Snapshot] = []
        self._denoise(prompt_ids, cfg, on_step=snapshots.append)
        return snapshots

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, cfg: TMASConfig) -> torch.Tensor:
        """Same sampler without instrumentation; returns the full sequence."""
        return self._denoise(prompt_ids, cfg, on_step=None)

    @torch.no_grad()
    def _denoise(
        self,
        prompt_ids: torch.Tensor,
        cfg: TMASConfig,
        on_step=None,
    ) -> torch.Tensor:
        """Block-wise low-confidence remasking, the reference LLaDA sampler.

        ``on_step`` receives a :class:`Snapshot` of the state the network just
        saw, before any token is committed -- that is what TMAS samples from.
        """
        device = next(self.model.parameters()).device
        prompt_ids = prompt_ids.to(device)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        p_len = prompt_ids.shape[1]
        total_len = p_len + cfg.gen_length
        x = torch.full(
            (1, total_len), self.mask_id, dtype=torch.long, device=device
        )
        x[:, :p_len] = prompt_ids

        if cfg.gen_length % cfg.block_length != 0:
            raise ValueError("gen_length must be divisible by block_length")
        n_blocks = cfg.gen_length // cfg.block_length
        if cfg.steps % n_blocks != 0:
            raise ValueError("steps must be divisible by the number of blocks")
        steps_per_block = cfg.steps // n_blocks

        for block_idx in range(n_blocks):
            lo = p_len + block_idx * cfg.block_length
            hi = p_len + (block_idx + 1) * cfg.block_length
            block_mask = x[:, lo:hi] == self.mask_id
            budget = get_num_transfer_tokens(block_mask, steps_per_block)

            for step in range(steps_per_block):
                mask_index = x == self.mask_id
                logits = self.model(x).logits

                probs = torch.softmax(logits.to(torch.float32), dim=-1)
                conf_all, x0 = probs.max(dim=-1)

                # Positions outside the current block are not eligible yet.
                conf = torch.where(mask_index, conf_all, torch.zeros_like(conf_all))
                eligible = torch.zeros_like(mask_index)
                eligible[:, lo:hi] = True
                selectable = mask_index & eligible

                # ---- snapshot the state the network just saw ----
                if on_step is not None:
                    resp_mask = mask_index[0, p_len:]
                    on_step(
                        Snapshot(
                            input_ids=x[0].detach().cpu().clone(),
                            mask=mask_index[0].detach().cpu().clone(),
                            confidence=conf[0].detach().float().cpu().clone(),
                            step=block_idx * steps_per_block + step,
                            total_steps=cfg.steps,
                            block_idx=block_idx,
                            mask_ratio=float(resp_mask.float().mean()),
                        )
                    )

                # ---- commit the most confident tokens of this block ----
                k = int(budget[0, step])
                if k > 0:
                    score = torch.where(
                        selectable, conf_all, torch.full_like(conf_all, -float("inf"))
                    )
                    idx = torch.topk(score[0], k=min(k, int(selectable.sum()))).indices
                    x[0, idx] = x0[0, idx]

        return x

    # -------------------------------------------------------------- rotation

    def rotation_plan(self):
        """Which modules take part in the QuaRot residual-stream rotation."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement rotation_plan(); "
            "run without --rotate or add it"
        )

    def down_projections(self) -> List[nn.Linear]:
        """MLP down-projections, the only place needing an online Hadamard."""
        out = []
        for block in self.blocks:
            for name, module in block.named_modules():
                if (
                    name.split(".")[-1] in ("ff_out", "down_proj", "w2")
                    and isinstance(module, nn.Linear)
                ):
                    out.append(module)
        return out

    def install_online_hadamards(self) -> None:
        """Wrap each down-projection so its input is rotated at runtime."""
        from ..algos.quarot import OnlineHadamard

        for block in self.blocks:
            for parent_name, parent in list(block.named_modules()):
                for attr, child in list(parent.named_children()):
                    if attr in ("ff_out", "down_proj", "w2") and isinstance(
                        child, nn.Linear
                    ):
                        setattr(parent, attr, OnlineHadamard(child))

    # ------------------------------------------------------------- utilities

    def encode_prompts(self, texts: Sequence[str], max_len: int) -> List[torch.Tensor]:
        out = []
        for t in texts:
            ids = self.tokenizer(t, return_tensors="pt", truncation=True,
                                 max_length=max_len)["input_ids"][0]
            out.append(ids)
        return out

    def describe(self) -> str:
        name, blocks = discover_blocks(self.model)
        return (
            f"{type(self).__name__}: {len(blocks)} blocks at '{name}', "
            f"mask_id={self.mask_id}"
        )


class AttentionProbe(ABC):
    """Captures the value matrix and softmax weights of one attention layer."""

    def __init__(self, module: nn.Module):
        self.module = module
        self.parts: Optional[AttentionParts] = None
        self._handle = None

    @abstractmethod
    def _hook(self, module, args, kwargs, output) -> None: ...

    def __enter__(self) -> "AttentionProbe":
        self._handle = self.module.register_forward_hook(
            self._hook, with_kwargs=True
        )
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.parts = None


__all__ = [
    "ModelAdapter",
    "AttentionProbe",
    "AttentionParts",
    "ArchitectureMismatch",
    "discover_blocks",
    "find_submodule",
    "get_num_transfer_tokens",
]
