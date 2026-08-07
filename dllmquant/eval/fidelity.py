"""How much did quantization change what the model actually predicts?

GSM8K is the metric that matters, but it costs a full generation per problem
and only reports a single number at the end.  This module measures the damage
directly and cheaply: take held-out states from a real denoising trajectory,
compare the FP16 model's predictions at masked positions with the quantized
model's, and report the disagreement *per mask ratio*.

That last part is the diffusion-specific bit.  A single aggregate hides the
thing worth knowing -- whether the loss is concentrated at the start of the
trajectory (where errors propagate through every remaining step) or at the end
(where tokens are being committed).  Two recipes with the same GSM8K score can
fail in completely different places, and the fix differs.

Three metrics, all computed on masked positions only, because the sampler
discards predictions at decoded ones:

  top-1 agreement  fraction of positions where the argmax is unchanged --
                   the tokens the sampler would actually commit
  KL divergence    KL(P_fp16 || P_quant), sensitive to damage that has not yet
                   flipped the argmax but is about to
  confidence drift mean change in max probability -- a systematic drop means
                   remasking will commit tokens in a different order
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from ..calib.tmas import Snapshot, bucket_index


@dataclass
class FidelityResult:
    """Metrics per mask-ratio bucket, plus the aggregate."""

    per_bucket: Dict[str, Dict[str, float]] = field(default_factory=dict)
    overall: Dict[str, float] = field(default_factory=dict)
    n_positions: int = 0

    def summary(self) -> str:
        lines = [
            f"prediction fidelity on {self.n_positions} masked positions:",
            f"  overall: top-1 agreement {100 * self.overall['top1']:.2f}%, "
            f"KL {self.overall['kl']:.4f}, "
            f"confidence drift {self.overall['conf_delta']:+.4f}",
        ]
        for name in sorted(self.per_bucket, reverse=True):
            m = self.per_bucket[name]
            lines.append(
                f"  mask ratio {name}: top-1 {100 * m['top1']:.2f}%, "
                f"KL {m['kl']:.4f}, drift {m['conf_delta']:+.4f}  "
                f"(n={int(m['n'])})"
            )
        return "\n".join(lines)


class ReferenceProbes:
    """FP16 predictions captured before quantization, at masked positions.

    Only a bounded number of positions is kept per state: a full logit tensor
    is [L, 126464] and storing several of those would cost more memory than the
    model.
    """

    def __init__(self, max_positions_per_state: int = 64, seed: int = 0):
        self.max_positions = max_positions_per_state
        self.generator = torch.Generator().manual_seed(seed)
        self.states: List[torch.Tensor] = []
        self.positions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.mask_ratios: List[float] = []

    @torch.no_grad()
    def capture(self, model, snapshots: Sequence[Snapshot], device) -> "ReferenceProbes":
        for snap in snapshots:
            masked = torch.nonzero(snap.mask, as_tuple=False).flatten()
            if masked.numel() == 0:
                continue
            if masked.numel() > self.max_positions:
                pick = torch.randperm(masked.numel(), generator=self.generator)
                masked = masked[pick[: self.max_positions]]

            ids = snap.input_ids.unsqueeze(0).to(device)
            logits = model(ids).logits[0].float()
            lp = torch.log_softmax(logits[masked], dim=-1)

            self.states.append(snap.input_ids.clone())
            self.positions.append(masked.clone())
            self.log_probs.append(lp.cpu())
            self.mask_ratios.append(snap.mask_ratio)
        return self

    def __len__(self) -> int:
        return len(self.states)


@torch.no_grad()
def measure_fidelity(
    model,
    probes: ReferenceProbes,
    device,
    n_buckets: int = 4,
) -> FidelityResult:
    """Compare the current (quantized) model against captured FP16 references."""
    if len(probes) == 0:
        raise ValueError("no reference probes captured")

    agg = {"top1": 0.0, "kl": 0.0, "conf_delta": 0.0, "n": 0.0}
    buckets: Dict[str, Dict[str, float]] = {}

    for ids_cpu, pos, ref_lp, ratio in zip(
        probes.states, probes.positions, probes.log_probs, probes.mask_ratios
    ):
        ids = ids_cpu.unsqueeze(0).to(device)
        logits = model(ids).logits[0].float()
        cur_lp = torch.log_softmax(logits[pos.to(device)], dim=-1).cpu()

        ref_p = ref_lp.exp()
        top1 = (cur_lp.argmax(-1) == ref_lp.argmax(-1)).float()
        kl = (ref_p * (ref_lp - cur_lp)).sum(-1)
        drift = cur_lp.max(-1).values.exp() - ref_p.max(-1).values

        n = float(top1.numel())
        b = _bucket_name(ratio, n_buckets)
        slot = buckets.setdefault(
            b, {"top1": 0.0, "kl": 0.0, "conf_delta": 0.0, "n": 0.0}
        )
        for key, val in (
            ("top1", float(top1.sum())),
            ("kl", float(kl.sum())),
            ("conf_delta", float(drift.sum())),
        ):
            slot[key] += val
            agg[key] += val
        slot["n"] += n
        agg["n"] += n

    for slot in buckets.values():
        for key in ("top1", "kl", "conf_delta"):
            slot[key] /= max(slot["n"], 1.0)
    for key in ("top1", "kl", "conf_delta"):
        agg[key] /= max(agg["n"], 1.0)

    return FidelityResult(
        per_bucket=buckets,
        overall={k: agg[k] for k in ("top1", "kl", "conf_delta")},
        n_positions=int(agg["n"]),
    )


def _bucket_name(ratio: float, n_buckets: int) -> str:
    idx = bucket_index(ratio, n_buckets)
    hi = 1.0 - idx / n_buckets
    lo = 1.0 - (idx + 1) / n_buckets
    return f"{hi:.2f}-{lo:.2f}"


__all__ = ["ReferenceProbes", "measure_fidelity", "FidelityResult"]
