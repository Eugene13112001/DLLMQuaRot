"""Temporal-Mask Adaptive Sampling.

A DLLM does not have one activation distribution -- it has one per denoising
step.  Early steps run on a nearly all-[MASK] sequence, late steps on a nearly
fully decoded one, and a calibration set drawn from plain text sees neither.

TMAS rolls out real denoising trajectories, snapshots the intermediate states,
buckets them by mask ratio and draws the calibration set with a U-shaped
allocation (default 0.3 / 0.2 / 0.2 / 0.3) that over-samples the two extremes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import torch

from ..config import TMASConfig


@dataclass
class Snapshot:
    """One intermediate state of the reverse diffusion process."""

    input_ids: torch.Tensor  # [L] int64, contains mask_id at masked positions
    mask: torch.Tensor  # [L] bool, True where still masked
    confidence: torch.Tensor  # [L] float, model's max prob per position (0 if decoded)
    step: int
    total_steps: int
    block_idx: int
    mask_ratio: float  # fraction of *response* positions still masked

    def to(self, device) -> "Snapshot":
        return Snapshot(
            input_ids=self.input_ids.to(device),
            mask=self.mask.to(device),
            confidence=self.confidence.to(device),
            step=self.step,
            total_steps=self.total_steps,
            block_idx=self.block_idx,
            mask_ratio=self.mask_ratio,
        )


def bucket_index(mask_ratio: float, n_buckets: int) -> int:
    """Map a mask ratio to a bucket; bucket 0 = most masked (earliest)."""
    if not 0.0 <= mask_ratio <= 1.0:
        mask_ratio = min(max(mask_ratio, 0.0), 1.0)
    # ratio 1.0 -> bucket 0, ratio 0.0 -> bucket n-1
    idx = int((1.0 - mask_ratio) * n_buckets)
    return min(idx, n_buckets - 1)


def allocate(n_samples: int, proportions: Sequence[float]) -> List[int]:
    """Largest-remainder allocation so the counts sum exactly to n_samples."""
    raw = [n_samples * p for p in proportions]
    counts = [int(r) for r in raw]
    remainder = n_samples - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in range(remainder):
        counts[order[i % len(order)]] += 1
    return counts


class TMASSampler:
    """Buckets snapshots by mask ratio and draws the calibration set."""

    def __init__(self, cfg: TMASConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    def bucketize(self, snapshots: Iterable[Snapshot]) -> List[List[Snapshot]]:
        buckets: List[List[Snapshot]] = [[] for _ in range(self.cfg.n_buckets)]
        for s in snapshots:
            buckets[bucket_index(s.mask_ratio, self.cfg.n_buckets)].append(s)
        return buckets

    def select(self, snapshots: Sequence[Snapshot]) -> List[Snapshot]:
        """Draw ``cfg.n_samples`` snapshots following ``cfg.proportions``.

        Buckets that cannot meet their quota (a short trajectory may never land
        in one) donate the shortfall to the remaining buckets proportionally,
        so the returned set always has ``n_samples`` entries as long as enough
        snapshots exist overall.
        """
        if not snapshots:
            raise ValueError("no snapshots to sample from")

        if self.cfg.uniform:
            # Ablation baseline: inherit whatever mask-ratio distribution the
            # trajectory happens to have.
            k = min(self.cfg.n_samples, len(snapshots))
            return self.rng.sample(list(snapshots), k)

        buckets = self.bucketize(snapshots)
        quotas = allocate(self.cfg.n_samples, list(self.cfg.proportions))

        chosen: List[Snapshot] = []
        shortfall = 0
        leftovers: List[List[Snapshot]] = []

        for bucket, quota in zip(buckets, quotas):
            if len(bucket) >= quota:
                picked = self.rng.sample(bucket, quota)
                taken = set(id(p) for p in picked)
                leftovers.append([s for s in bucket if id(s) not in taken])
            else:
                picked = list(bucket)
                shortfall += quota - len(bucket)
                leftovers.append([])
            chosen.extend(picked)

        # Redistribute the shortfall over whatever is left, preserving order of
        # preference: the extreme buckets first, mirroring the U-shape.
        if shortfall:
            pool = [s for b in leftovers for s in b]
            self.rng.shuffle(pool)
            chosen.extend(pool[:shortfall])

        self.rng.shuffle(chosen)
        return chosen

    # -------------------------------------------------------------- reporting

    def describe(self, snapshots: Sequence[Snapshot]) -> str:
        buckets = self.bucketize(snapshots)
        lines = [f"TMAS: {len(snapshots)} snapshots -> {self.cfg.n_samples} samples"]
        for i, (b, q) in enumerate(zip(buckets, allocate(
            self.cfg.n_samples, list(self.cfg.proportions)
        ))):
            lo = 1.0 - (i + 1) / self.cfg.n_buckets
            hi = 1.0 - i / self.cfg.n_buckets
            ratios = [s.mask_ratio for s in b]
            mean = sum(ratios) / len(ratios) if ratios else float("nan")
            lines.append(
                f"  bucket {i} (mask ratio {hi:.2f}..{lo:.2f}): "
                f"{len(b):5d} available, quota {q:4d}, mean ratio {mean:.3f}"
            )
        return "\n".join(lines)


def build_calibration_set(
    trace_fn: Callable[[torch.Tensor], List[Snapshot]],
    prompts: Sequence[torch.Tensor],
    cfg: TMASConfig,
    verbose: bool = True,
) -> List[Snapshot]:
    """Roll out trajectories for ``prompts`` and return the TMAS selection.

    ``trace_fn`` maps a prompt token tensor to the list of snapshots produced
    while generating a response for it -- see
    :meth:`dllmquant.models.base.ModelAdapter.trace_generation`.
    """
    sampler = TMASSampler(cfg)
    all_snapshots: List[Snapshot] = []

    for i, prompt in enumerate(prompts[: cfg.n_prompts]):
        snaps = trace_fn(prompt)
        if cfg.snapshot_every > 1:
            snaps = snaps[:: cfg.snapshot_every]
        all_snapshots.extend(snaps)
        if verbose and (i + 1) % 8 == 0:
            print(f"  [TMAS] {i + 1}/{min(len(prompts), cfg.n_prompts)} prompts, "
                  f"{len(all_snapshots)} snapshots")

    if verbose:
        print(sampler.describe(all_snapshots))

    return sampler.select(all_snapshots)


def text_calibration_set(
    prompts: Sequence[torch.Tensor],
    cfg: TMASConfig,
    verbose: bool = True,
) -> List[Snapshot]:
    """Calibration on plain text, with no denoising involved.

    This is the baseline TMAS exists to beat, and reproducing the paper's
    QuaRot/AWQ rows honestly requires it: those methods were written for
    autoregressive LLMs and have no notion of a trajectory, so ported
    unchanged they see fully-decoded sequences and nothing else.

    Every position is marked decoded with zero confidence.  No special-casing
    is needed downstream -- CGQ's certainty weights then come out uniform, so
    the Hessian degenerates to vanilla GPTQ exactly as it should.
    """
    snapshots: List[Snapshot] = []
    for i, ids in enumerate(prompts):
        ids = ids.flatten()[: cfg.seq_len]
        if ids.numel() < 2:
            continue
        snapshots.append(
            Snapshot(
                input_ids=ids.clone(),
                mask=torch.zeros(ids.numel(), dtype=torch.bool),
                confidence=torch.zeros(ids.numel(), dtype=torch.float32),
                step=0,
                total_steps=1,
                block_idx=0,
                mask_ratio=0.0,
            )
        )
        if len(snapshots) >= cfg.n_samples:
            break

    if not snapshots:
        raise ValueError("no usable text prompts for calibration")
    if verbose:
        print(f"text calibration: {len(snapshots)} sequences, no masked positions")
    return snapshots


__all__ = [
    "Snapshot",
    "TMASSampler",
    "build_calibration_set",
    "text_calibration_set",
    "bucket_index",
    "allocate",
]
