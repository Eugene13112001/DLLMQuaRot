"""Configuration objects for DLLMQuant.

Mirrors the three components of the paper (arXiv:2508.14090):
  * TMAS  -- Temporal-Mask Adaptive Sampling  (calibration set construction)
  * IA-AQ -- Interaction-Aware Activation Quantization (attention value matrix)
  * CGQ   -- Certainty-Guided Quantization (mask/confidence weighted Hessian)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass
class QuantConfig:
    """Numeric format of a single quantizer."""

    n_bits: int = 4
    symmetric: bool = False
    # 'per_tensor'  : one scalar for the whole tensor
    # 'per_channel' : one scale per row (weights) / per token (activations)
    # 'per_token'   : dynamic, recomputed per forward
    # 'per_feature' : one scale per hidden channel, SHARED across tokens --
    #                 the only setting where token importance (IA-AQ) matters
    # 'per_group'   : one scale per group_size elements along the last axis
    granularity: str = "per_channel"
    group_size: int = -1
    # Recompute scale/zero from every tensor that flows through, instead of
    # freezing them at calibration.  Required for any ACTIVATION quantizer
    # other than per_token: a frozen per_group scale is shaped after the first
    # calibration batch and silently misapplies (or fails to broadcast) once
    # the batch or sequence length changes.  per_token is always dynamic.
    dynamic: bool = False
    # Fixed fraction of the observed range to keep.  QuaRot uses a constant
    # 0.9 for activations and 0.95 for the KV cache instead of searching --
    # cheap, and it costs little once rotation has removed the outliers that
    # made a search worthwhile.  Ignored when mse_search is on.
    clip_ratio: float = 1.0
    # Search clipping ratio by (weighted) MSE instead of plain min/max.
    mse_search: bool = False
    mse_candidates: int = 20
    # Lowest clip ratio explored by the MSE search.
    mse_min_ratio: float = 0.4

    def __post_init__(self) -> None:
        if self.n_bits < 2 or self.n_bits > 16:
            raise ValueError(f"n_bits must be in [2, 16], got {self.n_bits}")
        valid = {"per_tensor", "per_channel", "per_token", "per_feature", "per_group"}
        if self.granularity not in valid:
            raise ValueError(f"granularity must be one of {sorted(valid)}")
        if self.granularity == "per_group" and self.group_size <= 0:
            raise ValueError("per_group granularity requires group_size > 0")


@dataclass
class TMASConfig:
    """Temporal-Mask Adaptive Sampling.

    The denoising trajectory is split into ``n_buckets`` contiguous blocks by
    mask ratio (bucket 0 = most masked / earliest, bucket -1 = least masked /
    latest).  ``proportions`` is the share of the calibration set drawn from
    each bucket; the paper's U-shaped 0.3/0.2/0.2/0.3 allocation spends more
    budget on the two extremes, where the token distribution is most atypical.
    """

    n_samples: int = 128
    n_buckets: int = 4
    proportions: Sequence[float] = (0.3, 0.2, 0.2, 0.3)
    # Prompts used to roll out denoising trajectories.
    n_prompts: int = 32
    seq_len: int = 256
    gen_length: int = 128
    block_length: int = 32
    steps: int = 64
    # Snapshot every k-th denoising step (1 = every step).
    snapshot_every: int = 1
    # Ablation: draw the calibration set uniformly at random from the
    # trajectory instead of enforcing per-bucket quotas.  This is what a naive
    # implementation does, and the difference between it and the U-shaped
    # allocation *is* TMAS's contribution -- without this switch the first of
    # the paper's three components cannot be ablated at all.
    uniform: bool = False
    # 'trajectory' -- calibrate on real denoising states (TMAS, the method)
    # 'text'       -- calibrate on plain tokenized text with no masks at all,
    #                 which is what a PTQ method written for autoregressive
    #                 LLMs does when ported to a DLLM unchanged. Needed to
    #                 reproduce the paper's baselines faithfully; `uniform`
    #                 only removes the quotas, it still samples the trajectory.
    mode: str = "trajectory"
    seed: int = 0

    def __post_init__(self) -> None:
        if len(self.proportions) != self.n_buckets:
            raise ValueError(
                f"proportions has {len(self.proportions)} entries but "
                f"n_buckets={self.n_buckets}"
            )
        total = sum(self.proportions)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"proportions must sum to 1.0, got {total}")
        if any(p < 0 for p in self.proportions):
            raise ValueError("proportions must be non-negative")
        if self.mode not in {"trajectory", "text"}:
            raise ValueError(f"mode must be 'trajectory' or 'text', got {self.mode!r}")


@dataclass
class CGQConfig:
    """Certainty-Guided Quantization.

    Per-token weights entering the Hessian ``H = 2 * X diag(c) X^T``:

        c_j = unmasked_weight                       if token j is already decoded
        c_j = masked_base + beta * confidence_j     if token j is masked

    Decoded tokens are down-weighted (their contribution to the layer output is
    already fixed), while masked tokens the model is confident about -- the ones
    about to be committed in the next denoising steps -- dominate the error
    compensation.
    """

    unmasked_weight: float = 0.5
    masked_base: float = 0.5
    beta: float = 1.0
    # Normalise c to mean 1.0 so percdamp stays comparable to vanilla GPTQ.
    normalize: bool = True
    percdamp: float = 0.01
    blocksize: int = 128
    act_order: bool = True
    static_groups: bool = False

    def __post_init__(self) -> None:
        for name in ("unmasked_weight", "masked_base", "beta"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 < self.percdamp < 1.0:
            raise ValueError(f"percdamp must be in (0, 1), got {self.percdamp}")


@dataclass
class IAAQConfig:
    """Interaction-Aware Activation Quantization for the attention value matrix.

    Token importance is the column mass of the softmax matrix -- how much the
    rest of the sequence attends to that token.  The V quantizer's clipping
    ratio is chosen to minimise the *importance-weighted* MSE, so tokens that
    many queries read from keep their dynamic range.
    """

    enabled: bool = True
    n_bits: int = 4
    # Average attention over heads before computing column mass.
    reduce_heads: str = "mean"  # 'mean' | 'max'
    # Exponent on the importance weights; >1 sharpens the priority.
    weight_power: float = 1.0
    # Floor added to every weight, as a fraction of the *average* token weight,
    # so no token is discarded outright.  It bounds how large an outlier the
    # method is still willing to clip: squared error scales with magnitude^2,
    # so a token weighted `f` blocks clipping once its magnitude exceeds
    # ~1/sqrt(f) times the typical one.  1e-4 tolerates roughly 100x outliers.
    weight_floor: float = 1e-4
    # The clip search must reach far below the observed range: attention value
    # outliers run 10-100x the typical magnitude, and a search that stops at
    # 0.4 of full range never shrinks the step size enough to matter.
    n_candidates: int = 40
    min_ratio: float = 0.05

    def __post_init__(self) -> None:
        if self.reduce_heads not in {"mean", "max"}:
            raise ValueError("reduce_heads must be 'mean' or 'max'")
        if self.weight_power <= 0:
            raise ValueError("weight_power must be positive")


@dataclass
class RotationConfig:
    """QuaRot-style orthogonal rotations (arXiv:2404.00456).

    Not part of DLLMQuant as published -- the paper uses QuaRot only as a
    baseline to compare against.  It is off by default so the default recipe
    stays a faithful reproduction; turn it on to attack the one thing the
    paper's three components do not touch, namely channel-wise activation
    outliers.  IA-AQ prioritises along the token axis, rotation removes the
    need to prioritise along the channel axis.
    """

    # Numbering follows QuaRot's own paper:
    #   R1 global hidden-state rotation
    #   R2 online Hadamard before the FFN down-projection
    #   R3 head-wise rotation of the value projections
    #   R4 online head-wise rotation of Q and K after RoPE  (NOT implemented
    #      here -- it exists to make the KV cache quantizable, and this
    #      pipeline leaves Q, K and the softmax matrix in bf16)
    enabled: bool = False
    residual: bool = True       # R1
    value_heads: bool = True    # R3 -- the one that interacts with IA-AQ
    online_mlp: bool = True     # R2 -- part of QuaRot's recipe, so on by default
    qk_heads: bool = False      # R4 -- needs a patchable attention forward
    randomized: bool = True     # random sign flips, as in the paper
    seed: int = 0
    # Fail rather than quantize if rotation changed the model's output by more
    # than this (relative). Rotation is supposed to be exactly invariant.
    invariance_tol: float = 1e-2
    # Mask ratios at which invariance is verified. A DLLM has no single
    # activation regime, so checking one state proves little.
    check_mask_ratios: Sequence[float] = (1.0, 0.75, 0.5, 0.25, 0.0)


@dataclass
class DLLMQuantConfig:
    """Top-level recipe."""

    model_path: str = ""
    model_type: str = "llada"  # 'llada' | 'llada2_moe'
    weight: QuantConfig = field(
        default_factory=lambda: QuantConfig(
            n_bits=4, symmetric=False, granularity="per_channel", mse_search=True
        )
    )
    activation: QuantConfig = field(
        default_factory=lambda: QuantConfig(
            n_bits=4, symmetric=False, granularity="per_token"
        )
    )
    tmas: TMASConfig = field(default_factory=TMASConfig)
    cgq: CGQConfig = field(default_factory=CGQConfig)
    ia_aq: IAAQConfig = field(default_factory=IAAQConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)

    # Layers left in FP.  Two different matches, because substring matching
    # alone is a trap on these architectures:
    #   * LLaDA blocks call their MLP down-projection `ff_out`, and so does the
    #     top-level LM head -- a substring rule would skip every MLP.
    #   * `gate` (the MoE router, which must stay FP) is a substring of
    #     `gate_proj` (an ordinary MLP layer, which must not be skipped).
    # So routers are matched on the exact leaf name, paths on substrings.
    skip_leaf_names: List[str] = field(
        default_factory=lambda: ["gate", "router", "wg", "lm_head"]
    )
    skip_patterns: List[str] = field(
        default_factory=lambda: ["lm_head", "embed_tokens", "wte", "embeddings"]
    )
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 0

    def skip(self, name: str) -> bool:
        leaf = name.split(".")[-1]
        return leaf in self.skip_leaf_names or any(p in name for p in self.skip_patterns)


def w4a4() -> DLLMQuantConfig:
    """The paper's headline setting: 4-bit weights, 4-bit activations."""
    return DLLMQuantConfig()


def quarot_w4a4() -> DLLMQuantConfig:
    """DLLMQuant's three components on top of QuaRot's numeric recipe.

    Matches the settings QuaRot reports: symmetric per-token activations with a
    constant 0.9 clipping ratio (no search -- rotation has already removed the
    outliers a search would be chasing), weights solved with a squared-error
    clip search, and rotations R1/R2/R3 enabled.

    R4 stays off: it requires patching the attention forward, which the
    remote-code LLaDA modelling file does not expose (see RotationConfig).
    """
    cfg = DLLMQuantConfig()
    cfg.activation = QuantConfig(
        n_bits=4, symmetric=True, granularity="per_token", clip_ratio=0.9
    )
    cfg.ia_aq = IAAQConfig(enabled=True, n_bits=4)
    cfg.rotation = RotationConfig(
        enabled=True, residual=True, value_heads=True, online_mlp=True
    )
    return cfg


def quarot_baseline() -> DLLMQuantConfig:
    """QuaRot exactly as the paper uses it as a baseline -- nothing of DLLMQuant.

    Rotations R1/R2/R3, GPTQ for the weights, QuaRot's numeric settings, and
    plain text calibration.  All three of the paper's components are off:

      * calibration is text, not trajectory states   (no TMAS)
      * the value matrix is not quantized separately (no IA-AQ)
      * every token weighs the same in the Hessian   (CGQ degenerates to GPTQ)

    The third falls out on its own: text snapshots carry no masked positions,
    so the certainty weights come out uniform without any special-casing.
    """
    cfg = DLLMQuantConfig()
    cfg.activation = QuantConfig(
        n_bits=4, symmetric=True, granularity="per_token", clip_ratio=0.9
    )
    cfg.tmas = TMASConfig(mode="text")
    cfg.cgq = CGQConfig(unmasked_weight=1.0, masked_base=1.0, beta=0.0)
    cfg.ia_aq = IAAQConfig(enabled=False)
    cfg.rotation = RotationConfig(
        enabled=True, residual=True, value_heads=True, online_mlp=True
    )
    return cfg


def quarot_diffusion() -> DLLMQuantConfig:
    """QuaRot adapted to a diffusion LM -- and nothing from DLLMQuant.

    The rotations themselves need no adaptation: a random Hadamard never looks
    at data, so R1/R2/R3 are exactly as invariant on a masked diffusion model
    as on an autoregressive one.  What does not transfer is the *other* half of
    QuaRot: weights are solved by GPTQ, and GPTQ is calibration-dependent.

    So the single change from :func:`quarot_baseline` is the calibration
    source -- real denoising states instead of plain text.  Sampling stays
    uniform over the trajectory: drawing states at all is the minimal
    adaptation, while the U-shaped quota scheme is TMAS proper and belongs to
    DLLMQuant, not here.

    The ladder, each step adding exactly one thing:

        quarot_baseline()   rotations + GPTQ, text calibration
        quarot_diffusion()  ... calibrated on denoising states instead
        w4a4()              ... + TMAS quotas + IA-AQ + certainty weights
        quarot_w4a4()       ... + rotations again, all of it together
    """
    cfg = quarot_baseline()
    cfg.tmas = TMASConfig(mode="trajectory", uniform=True)
    return cfg


def w4a16() -> DLLMQuantConfig:
    """Weight-only 4-bit; activations stay FP16 and IA-AQ is inert."""
    cfg = DLLMQuantConfig()
    cfg.activation = QuantConfig(n_bits=16, granularity="per_token")
    cfg.ia_aq = IAAQConfig(enabled=False)
    return cfg


__all__ = [
    "QuantConfig",
    "TMASConfig",
    "CGQConfig",
    "IAAQConfig",
    "RotationConfig",
    "DLLMQuantConfig",
    "w4a4",
    "quarot_w4a4",
    "quarot_baseline",
    "quarot_diffusion",
    "w4a16",
]
