"""Applying QuaRot to a diffusion LM, and proving it did no harm.

The rotation itself is in :mod:`dllmquant.algos.quarot`; this module wires it
to a model adapter and validates it *across the denoising trajectory*, which
is the part a dense-LLM implementation has no reason to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .algos.quarot import (
    RotationPlan,
    fuse_norm_into_linears,
    hadamard_factors,
    install_online_hadamard,
    random_hadamard_matrix,
    rotate_residual_stream,
    rotate_value_heads,
)
from .config import DLLMQuantConfig
from .models.base import ModelAdapter


@dataclass
class RotationReport:
    d_model: int
    head_dim: int
    blocks: int
    norms_fused: int
    online_mlp: int
    # mask ratio -> metrics
    invariance: Dict[str, float] = field(default_factory=dict)
    outliers_before: Dict[str, float] = field(default_factory=dict)
    outliers_after: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"QuaRot: d_model={self.d_model}, {self.blocks} blocks, "
            f"{self.norms_fused} norms fused, {self.online_mlp} online Hadamards"
        ]
        for key in sorted(self.invariance, key=float, reverse=True):
            before = self.outliers_before.get(key, float("nan"))
            after = self.outliers_after.get(key, float("nan"))
            lines.append(
                f"  mask ratio {key}: rel. output change {self.invariance[key]:.2e}, "
                f"outlier factor {before:.1f} -> {after:.1f}"
            )
        return "\n".join(lines)


def crest_factor(x: torch.Tensor) -> float:
    """max|x| / rms(x) per token, averaged.

    A blunt but honest outlier metric: a tensor whose energy sits in one
    channel has a large crest factor, a rotated one approaches sqrt(2/pi)-ish
    Gaussian behaviour.  Reported before and after so the rotation's effect is
    a measurement, not a claim.
    """
    f = x.reshape(-1, x.shape[-1]).to(torch.float32)
    rms = f.pow(2).mean(dim=-1).sqrt().clamp(min=1e-8)
    return float((f.abs().amax(dim=-1) / rms).mean())


@torch.no_grad()
def apply_quarot(adapter: ModelAdapter, cfg: DLLMQuantConfig) -> RotationReport:
    """Fuse norms, rotate the model, and verify it still computes the same thing.

    Must run BEFORE calibration: CGQ's Hessians and IA-AQ's value scales are
    only valid in the basis the model will actually run in.
    """
    rot = cfg.rotation
    plan = adapter.rotation_plan()
    d_model = adapter.n_heads * adapter.head_dim
    device = next(adapter.model.parameters()).device

    report = RotationReport(
        d_model=d_model,
        head_dim=adapter.head_dim,
        blocks=len(adapter.blocks),
        norms_fused=0,
        online_mlp=0,
    )

    # --- reference behaviour, sampled across the trajectory -----------------
    states = _trajectory_states(adapter, cfg)
    reference = {k: adapter.model(v).logits.float().clone() for k, v in states.items()}
    for k, v in states.items():
        report.outliers_before[k] = crest_factor(_hidden_at_block0(adapter, v))

    # --- fuse norms so rotation commutes ------------------------------------
    for norm, consumers in plan.norm_groups:
        fuse_norm_into_linears(norm, consumers)
        report.norms_fused += 1

    # --- R1: residual stream -------------------------------------------------
    if rot.residual:
        q = random_hadamard_matrix(
            d_model, device=device, dtype=torch.float32, seed=rot.seed
        ) if rot.randomized else None
        if q is None:
            from .algos.quarot import hadamard_matrix

            q = hadamard_matrix(d_model, device=device, dtype=torch.float32)
        rotate_residual_stream(plan, q)

    # --- R3: per-head value rotation ----------------------------------------
    if rot.value_heads:
        h = random_hadamard_matrix(
            adapter.head_dim, device=device, dtype=torch.float32, seed=rot.seed + 1
        )
        for v_proj, out_proj, v_offset in plan.head_pairs:
            rotate_value_heads(
                v_proj, out_proj, adapter.n_heads, adapter.head_dim, h, v_offset
            )

    # --- R2: online Hadamard in front of the MLP down-projection -------------
    if rot.online_mlp:
        for down in adapter.down_projections():
            if hadamard_factors(down.in_features) is None:
                continue
            install_online_hadamard(down)
            report.online_mlp += 1
        adapter.install_online_hadamards()

    # --- verification --------------------------------------------------------
    for k, v in states.items():
        after = adapter.model(v).logits.float()
        denom = reference[k].abs().mean().clamp(min=1e-6)
        report.invariance[k] = float((after - reference[k]).abs().mean() / denom)
        report.outliers_after[k] = crest_factor(_hidden_at_block0(adapter, v))

    worst = max(report.invariance.values()) if report.invariance else 0.0
    if worst > rot.invariance_tol:
        raise RuntimeError(
            f"rotation changed the model's output by {worst:.3e} (tolerance "
            f"{rot.invariance_tol:.1e}). Rotation must be exactly invariant; "
            "this means a layer that reads or writes the residual stream was "
            "missed by rotation_plan(), or a norm has a bias term.\n"
            + report.summary()
        )
    return report


def _trajectory_states(
    adapter: ModelAdapter, cfg: DLLMQuantConfig
) -> Dict[str, torch.Tensor]:
    """One input per requested mask ratio.

    Invariance is algebraic and should hold everywhere, but the *numerics* are
    not uniform: at mask ratio 1.0 nearly every position carries the same
    embedding row, and a rotation that quietly missed a layer shows up there
    first.
    """
    device = next(adapter.model.parameters()).device
    prompt = adapter.encode_prompts(["rotation invariance probe sequence"], max_len=16)[0]
    p_len = prompt.shape[-1]
    gen = cfg.tmas.block_length * 2

    out: Dict[str, torch.Tensor] = {}
    filler = adapter.encode_prompts(["filler tokens for the decoded region"],
                                    max_len=gen)[0]
    for ratio in cfg.rotation.check_mask_ratios:
        x = torch.full((1, p_len + gen), adapter.mask_id, dtype=torch.long)
        x[0, :p_len] = prompt
        n_decoded = int(round((1.0 - ratio) * gen))
        for i in range(n_decoded):
            x[0, p_len + i] = int(filler[i % filler.numel()])
        out[f"{ratio:.2f}"] = x.to(device)
    return out


def _hidden_at_block0(adapter: ModelAdapter, ids: torch.Tensor) -> torch.Tensor:
    """Capture the residual stream entering the first block."""
    grabbed: List[torch.Tensor] = []

    def hook(module, args, kwargs, output):
        h = kwargs.get("hidden_states", args[0] if args else None)
        if isinstance(h, torch.Tensor):
            grabbed.append(h.detach())

    handle = adapter.blocks[0].register_forward_hook(hook, with_kwargs=True)
    try:
        adapter.model(ids)
    finally:
        handle.remove()
    if not grabbed:
        raise RuntimeError("could not capture the block-0 input")
    return grabbed[0]


__all__ = ["apply_quarot", "RotationReport", "crest_factor"]
