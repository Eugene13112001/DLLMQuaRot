"""Comparing what the quantized model actually generates, not what it would
predict if it were held on the right path.

``eval/fidelity.py`` scores the quantized model on states taken from the FP16
model's trajectory.  That is teacher forcing: at every step the model is put
back on the correct path, so error never compounds.  On this project it read
88.9% for a configuration that solved 26% of GSM8K and 91.2% for one that
solved 78% -- a 2-point gap standing in for a 52-point one.

The reason is specific to diffusion.  A committed token is never revised, and
the order of commitment follows confidence, so a small confidence shift changes
*which* token gets frozen first; from there the model walks its own trajectory
into somewhere the reference never visits.  Teacher forcing cannot see that,
because it never lets the model walk.

This module lets it walk.  Both models generate from the same prompt under the
same deterministic sampler, and the two trajectories are compared: where they
first part, how far apart they end, and at what mask ratio the split happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from ..calib.tmas import Snapshot
from ..config import TMASConfig
from ..models.base import ModelAdapter


@dataclass
class ReferenceTrajectory:
    """One full FP16 generation, kept so a later run can be compared to it."""

    prompt_ids: torch.Tensor
    final_ids: torch.Tensor
    prompt_len: int
    # State the sampler held at each step, before committing that step.
    states: List[torch.Tensor] = field(default_factory=list)
    mask_ratios: List[float] = field(default_factory=list)


@dataclass
class TrajectoryResult:
    n_prompts: int = 0
    exact_match: float = 0.0
    final_agreement: float = 0.0
    mean_divergence_step: float = 0.0
    mean_divergence_ratio: float = 0.0
    never_diverged: int = 0
    per_prompt: List[dict] = field(default_factory=list)

    def summary(self) -> str:
        return "\n".join([
            f"trajectory divergence over {self.n_prompts} prompts:",
            f"  identical output      {100 * self.exact_match:.1f}% of prompts",
            f"  final token agreement {100 * self.final_agreement:.2f}%",
            f"  first divergence at step {self.mean_divergence_step:.1f} "
            f"(mask ratio {self.mean_divergence_ratio:.2f})",
            f"  never diverged        {self.never_diverged}/{self.n_prompts}",
        ])


@torch.no_grad()
def capture_reference(
    adapter: ModelAdapter,
    prompts: Sequence[torch.Tensor],
    cfg: TMASConfig,
    verbose: bool = False,
) -> List[ReferenceTrajectory]:
    """Record the FP16 model's own generations.  Run BEFORE quantizing."""
    refs: List[ReferenceTrajectory] = []
    for i, prompt in enumerate(prompts):
        states: List[torch.Tensor] = []
        ratios: List[float] = []

        def on_step(s: Snapshot) -> None:
            states.append(s.input_ids.clone())
            ratios.append(s.mask_ratio)

        final = adapter._denoise(prompt, cfg, on_step=on_step)
        refs.append(
            ReferenceTrajectory(
                prompt_ids=prompt.clone(),
                final_ids=final[0].detach().cpu().clone(),
                prompt_len=int(prompt.flatten().shape[0]),
                states=states,
                mask_ratios=ratios,
            )
        )
        if verbose and (i + 1) % 4 == 0:
            print(f"  [trajectory] captured {i + 1}/{len(prompts)}")
    return refs


@torch.no_grad()
def measure_divergence(
    adapter: ModelAdapter,
    references: Sequence[ReferenceTrajectory],
    cfg: TMASConfig,
    verbose: bool = False,
) -> TrajectoryResult:
    """Let the current (quantized) model generate, and compare trajectories."""
    if not references:
        raise ValueError("no reference trajectories captured")

    result = TrajectoryResult(n_prompts=len(references))
    exact = 0
    agreements: List[float] = []
    div_steps: List[int] = []
    div_ratios: List[float] = []

    for i, ref in enumerate(references):
        states: List[torch.Tensor] = []

        def on_step(s: Snapshot) -> None:
            states.append(s.input_ids.clone())

        final = adapter._denoise(ref.prompt_ids, cfg, on_step=on_step)
        final = final[0].detach().cpu()

        # Where the two trajectories first hold different states. A state is
        # the sequence the sampler is looking at, so this is the step at which
        # a different token got committed.
        first_div = -1
        div_ratio = float("nan")
        for step, (a, b) in enumerate(zip(ref.states, states)):
            if a.shape != b.shape or not torch.equal(a, b):
                first_div = step
                div_ratio = (
                    ref.mask_ratios[step] if step < len(ref.mask_ratios) else float("nan")
                )
                break

        resp = slice(ref.prompt_len, None)
        same = (final[resp] == ref.final_ids[resp])
        agreement = float(same.float().mean()) if same.numel() else 1.0
        identical = bool(same.all())

        exact += int(identical)
        agreements.append(agreement)
        if first_div >= 0:
            div_steps.append(first_div)
            if div_ratio == div_ratio:  # not NaN
                div_ratios.append(div_ratio)
        else:
            result.never_diverged += 1

        result.per_prompt.append({
            "final_agreement": agreement,
            "identical": identical,
            "first_divergence_step": first_div,
            "divergence_mask_ratio": div_ratio,
        })
        if verbose and (i + 1) % 4 == 0:
            print(f"  [trajectory] compared {i + 1}/{len(references)}")

    result.exact_match = exact / len(references)
    result.final_agreement = sum(agreements) / len(agreements)
    result.mean_divergence_step = (
        sum(div_steps) / len(div_steps) if div_steps else float("nan")
    )
    result.mean_divergence_ratio = (
        sum(div_ratios) / len(div_ratios) if div_ratios else float("nan")
    )
    return result


__all__ = [
    "ReferenceTrajectory",
    "TrajectoryResult",
    "capture_reference",
    "measure_divergence",
]
