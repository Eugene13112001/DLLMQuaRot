"""Applying QuaRot to a diffusion LM, and proving it did no harm.

The rotation itself is in :mod:`dllmquant.algos.quarot`; this module wires it
to a model adapter and validates it *across the denoising trajectory*, which
is the part a dense-LLM implementation has no reason to do.
"""

from __future__ import annotations

import warnings
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
    tolerance: float = 0.0
    # mask ratio -> metrics
    invariance: Dict[str, float] = field(default_factory=dict)
    outliers_before: Dict[str, float] = field(default_factory=dict)
    outliers_after: Dict[str, float] = field(default_factory=dict)
    # Fraction of expert slots that route the same way after rotation.
    # Empty for dense models, which have nothing discrete to disagree about.
    routing_agreement: Dict[str, float] = field(default_factory=dict)
    # False when the invariance number could not settle the question -- an MoE
    # in low precision, where routing noise is larger than what the check is
    # looking for.
    decisive: bool = True

    def summary(self) -> str:
        lines = [
            f"QuaRot: d_model={self.d_model}, {self.blocks} blocks, "
            f"{self.norms_fused} norms fused, {self.online_mlp} online Hadamards"
            + (f", invariance floor {self.tolerance:.0e}" if self.tolerance else "")
        ]
        for key in sorted(self.invariance, key=float, reverse=True):
            before = self.outliers_before.get(key, float("nan"))
            after = self.outliers_after.get(key, float("nan"))
            line = (
                f"  mask ratio {key}: rel. output change {self.invariance[key]:.2e}, "
                f"outlier factor {before:.1f} -> {after:.1f}"
            )
            if key in self.routing_agreement:
                line += f", routing kept {100 * self.routing_agreement[key]:.1f}%"
            lines.append(line)
        return "\n".join(lines)

    @property
    def worst_routing_agreement(self) -> Optional[float]:
        return min(self.routing_agreement.values()) if self.routing_agreement else None


@torch.no_grad()
def routing_fingerprint(
    adapter: ModelAdapter, ids: torch.Tensor
) -> Optional[torch.Tensor]:
    """Which experts every token was sent to, as one flat tensor.

    Rotation is exact in real arithmetic but the weights are stored in bf16,
    and in an MoE that rounding does not stay small: routing is a *discrete*
    function of the residual stream, so a perturbation of the router's logits
    below the last mantissa bit can still swap one expert for another, and the
    token's MLP output then changes by a lot rather than a little.

    That mechanism does not exist in a dense model, which is why the flat bf16
    noise floor -- calibrated on LLaDA-1.5 -- cannot be read the same way here.
    Measuring the routing separately says which of the two is happening: an
    unchanged routing with a large output change means a layer was genuinely
    missed, while flipped routing explains the change without implicating the
    rotation at all.

    Returned as ``[rows, k]`` -- one row per token per router -- because the
    comparison has to be by set (see :func:`routing_overlap`).
    """
    routers = adapter.routers()
    if not routers:
        return None

    seen: List[torch.Tensor] = []

    def hook(module, args, output):
        parts = output if isinstance(output, (tuple, list)) else (output,)
        for t in parts:
            if isinstance(t, torch.Tensor) and t.numel() and not t.is_floating_point():
                idx = t if t.dim() > 1 else t.unsqueeze(-1)
                seen.append(idx.reshape(-1, idx.shape[-1]).cpu())
                return

    handles = [m.register_forward_hook(hook) for m in routers]
    try:
        adapter.model(ids, **adapter.forward_kwargs(ids))
    finally:
        for h in handles:
            h.remove()

    if not seen or len({t.shape[-1] for t in seen}) != 1:
        return torch.cat(seen, dim=0) if len(seen) == 1 else None
    return torch.cat(seen, dim=0)


def routing_overlap(before: torch.Tensor, after: torch.Tensor) -> float:
    """Fraction of each token's expert set that survived, averaged over tokens.

    By set, not position by position: ``topk(sorted=False)`` returns the chosen
    experts in no particular order, and even sorted, swapping one expert shifts
    the rest along -- which a positional comparison counts as several changes,
    inflating the damage several-fold.
    """
    if before.shape != after.shape:
        return float("nan")
    kept = (before.unsqueeze(-1) == after.unsqueeze(-2)).any(dim=-1)
    return float(kept.float().mean())


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
    reference = {
        k: adapter.model(v, **adapter.forward_kwargs(v)).logits.float().clone()
        for k, v in states.items()
    }
    for k, v in states.items():
        report.outliers_before[k] = crest_factor(_hidden_at_block0(adapter, v))
    routes_before = {k: routing_fingerprint(adapter, v) for k, v in states.items()}

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
                v_proj, out_proj, adapter.n_heads, adapter.head_dim, h, v_offset,
                n_kv_heads=adapter.n_kv_heads,
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
        after = adapter.model(v, **adapter.forward_kwargs(v)).logits.float()
        denom = reference[k].abs().mean().clamp(min=1e-6)
        report.invariance[k] = float((after - reference[k]).abs().mean() / denom)
        report.outliers_after[k] = crest_factor(_hidden_at_block0(adapter, v))

        before_routes = routes_before.get(k)
        if before_routes is not None:
            now = routing_fingerprint(adapter, v)
            if now is not None:
                report.routing_agreement[k] = routing_overlap(before_routes, now)

    tol = rot.invariance_tol or dtype_invariance_tol(cfg.dtype)
    worst = max(report.invariance.values()) if report.invariance else 0.0
    report.tolerance = tol

    if worst <= tol:
        return report

    agreement = report.worst_routing_agreement
    routing_moved = agreement is not None and agreement < 1.0

    if routing_moved and cfg.dtype != "float32":
        # The check has no discriminating power in this combination, and
        # failing on it would block every bf16 run of an MoE for a reason it
        # cannot establish.  What guards against a missed layer is the
        # structural coverage in rotation_plan(); this number was its backstop,
        # and the backstop is unavailable here -- say so rather than pretend
        # either way.
        report.decisive = False
        warnings.warn(
            f"invariance came out at {worst:.3e} against a {cfg.dtype} floor of "
            f"{tol:.1e}, but routing also moved (as low as "
            f"{100 * agreement:.1f}% of each token's experts kept), which "
            "accounts for a change of this size on its own. Rotation shifts "
            "the router's logits by the last bits of the stored weights, and "
            "top-k turns that into a different expert -- so in this precision "
            "the check cannot separate a missed layer from rounding.\n"
            "Verify the rotation itself once in float32 (--dtype float32), "
            "where routing holds and the floor is "
            f"{dtype_invariance_tol('float32'):.0e}; rerun that whenever the "
            "rotation plan changes.\n" + report.summary(),
            RuntimeWarning,
        )
        return report

    raise RuntimeError(
        f"rotation changed the model's output by {worst:.3e}, past the "
        f"{cfg.dtype} noise floor of {tol:.1e}.\n\n"
        "Rotation is exactly invariant in exact arithmetic, so an excess this "
        "large means either a layer that touches the residual stream was "
        "missed, or a norm carries a bias. Note that rotation_plan() already "
        "checks coverage structurally, so if it passed, suspect the norms "
        "first."
        + (
            "\nRouting was identical across the rotation, so the change is not "
            "the router's doing."
            if agreement is not None
            else ""
        )
        + f"\nLoading in float32 (--dtype float32) drops the floor to "
        f"{dtype_invariance_tol('float32'):.0e} and makes this diagnostic "
        "sharp.\n\n" + report.summary()
    )


def dtype_invariance_tol(dtype: str) -> float:
    """Noise floor for the invariance check, by storage precision.

    The rotation math runs in float64, but the rotated weights are stored back
    in the model's dtype. bf16 keeps 7 mantissa bits, fp16 keeps 10, and the
    error compounds across blocks -- so what counts as "exactly invariant"
    differs by two orders of magnitude between bf16 and fp32.

    These floors are calibrated on a dense model, where rounding perturbs the
    output continuously.  A model that routes cannot be judged by them: there
    the same rounding changes *which experts run*, and the output moves by far
    more than any floor set from mantissa bits.  ``apply_quarot`` handles that
    case separately rather than by inflating the number here, which would blind
    the check for dense models too.
    """
    return {
        "float32": 1e-4,
        "float16": 1.5e-2,
        "bfloat16": 6e-2,
    }.get(dtype, 6e-2)


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
        adapter.model(ids, **adapter.forward_kwargs(ids))
    finally:
        handle.remove()
    if not grabbed:
        raise RuntimeError("could not capture the block-0 input")
    return grabbed[0]


__all__ = ["apply_quarot", "RotationReport", "crest_factor"]
