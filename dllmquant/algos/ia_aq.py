"""Interaction-Aware Activation Quantization for the attention value matrix.

Per-token dynamic activation quantization gives every token its own scale, so
no token can hurt another and importance weighting would be meaningless.  The
value matrix is different: V is consumed as ``A @ V``, a *mixture* over tokens,
and a single shared scale must cover all of them.  Whichever token holds the
largest magnitude sets the step size for everybody.

IA-AQ picks that shared scale by minimising the error weighted by how much the
sequence actually attends to each token -- the column mass of the softmax
matrix.  A token nobody reads is allowed to clip; a token every query reads
keeps its range.

The attention matrix is only needed during calibration.  The resulting scale is
frozen into the quantizer, so inference costs nothing extra.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ..config import IAAQConfig, QuantConfig
from ..quantizers import InteractionAwareQuantizer


def interaction_weights(
    attn_probs: torch.Tensor,
    cfg: IAAQConfig,
    query_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Token importance from a softmax attention tensor.

    ``attn_probs`` is [B, heads, Q, K] with rows summing to 1.  Returns [B, K],
    normalised to mean 1 per sequence so the values are comparable across
    layers and batch sizes.

    ``query_mask`` is [B, Q], True where the position is still masked.  Rows on
    decoded positions are scaled by ``cfg.decoded_query_weight`` rather than
    dropped -- their logits are discarded only at the final layer, while
    everywhere below they still feed the masked positions through the layer
    above.
    """
    if attn_probs.dim() != 4:
        raise ValueError(f"expected [B, H, Q, K], got {tuple(attn_probs.shape)}")

    a = attn_probs.to(torch.float32)
    a = a.mean(dim=1) if cfg.reduce_heads == "mean" else a.amax(dim=1)  # [B, Q, K]

    if query_mask is not None:
        # Soft, not binary: decoded rows still reach the read-out positions
        # through the layers above, just weakly.
        row_w = torch.where(
            query_mask.to(torch.bool),
            torch.ones_like(a[:, :, 0]),
            torch.full_like(a[:, :, 0], cfg.decoded_query_weight),
        )
        a = a * row_w.unsqueeze(-1)
        denom = row_w.sum(dim=1, keepdim=True).clamp(min=1e-6)
    else:
        denom = torch.full(
            (a.shape[0], 1), float(a.shape[1]), device=a.device, dtype=a.dtype
        )

    imp = a.sum(dim=1) / denom  # [B, K], mean attention received

    if cfg.weight_power != 1.0:
        imp = imp.clamp(min=0).pow(cfg.weight_power)

    def _to_mean_one(t: torch.Tensor) -> torch.Tensor:
        return t * (t.shape[-1] / t.sum(dim=-1, keepdim=True).clamp(min=1e-8))

    # Normalise BEFORE adding the floor.  Raw column mass averages 1/K, so an
    # absolute floor would mean something different at every sequence length --
    # at K=128 a floor of 1e-3 is already 6% of the average token's weight,
    # enough to make the objective refuse to clip a 50x outlier.
    imp = _to_mean_one(imp)
    imp = imp + cfg.weight_floor
    return _to_mean_one(imp)


class InteractionCollector:
    """Accumulates (V, importance) pairs for one attention layer."""

    def __init__(self, cfg: IAAQConfig, max_tokens: int = 65536):
        self.cfg = cfg
        self.max_tokens = max_tokens
        self.values: List[torch.Tensor] = []
        self.weights: List[torch.Tensor] = []
        self.used_query_mask = False
        self._n_tokens = 0

    def add(
        self,
        value_states: torch.Tensor,
        attn_probs: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """``value_states``: [B, heads, K, head_dim] or [B, K, D].

        ``query_mask`` is [B, Q] and True where the query position is still
        masked; attention from decoded positions is then ignored, because the
        sampler throws those outputs away.
        """
        if self._n_tokens >= self.max_tokens:
            return

        if query_mask is not None and self.cfg.decoded_query_weight >= 1.0:
            # Decoded rows count fully: the mask would change nothing.
            query_mask = None

        v = value_states
        if v.dim() == 4:  # merge heads back into the feature axis
            b, h, k, d = v.shape
            v = v.permute(0, 2, 1, 3).reshape(b, k, h * d)

        w = interaction_weights(attn_probs, self.cfg, query_mask)
        self.used_query_mask = self.used_query_mask or query_mask is not None
        if w.shape[1] != v.shape[1]:
            raise ValueError(
                f"attention keys ({w.shape[1]}) != value tokens ({v.shape[1]})"
            )

        self.values.append(v.detach().to(torch.float32).cpu())
        self.weights.append(w.detach().to(torch.float32).cpu())
        self._n_tokens += v.shape[0] * v.shape[1]

    def build_quantizer(self, device=None) -> InteractionAwareQuantizer:
        """Calibrate and freeze the V quantizer on everything collected."""
        if not self.values:
            raise RuntimeError("no attention statistics collected")

        v = torch.cat([x.reshape(-1, x.shape[-1]) for x in self.values], dim=0)
        w = torch.cat([x.reshape(-1) for x in self.weights], dim=0)

        q_cfg = QuantConfig(
            n_bits=self.cfg.n_bits,
            symmetric=False,
            granularity="per_feature",
            mse_search=True,
            mse_candidates=self.cfg.n_candidates,
            mse_min_ratio=self.cfg.min_ratio,
        )
        quantizer = InteractionAwareQuantizer(q_cfg)
        if device is not None:
            v, w = v.to(device), w.to(device)
            quantizer = quantizer.to(device)

        quantizer.find_params(v, weights=w)
        quantizer.freeze()
        return quantizer

    def error_report(self, quantizer: InteractionAwareQuantizer) -> Dict[str, float]:
        """Weighted / unweighted MSE, for ablations against plain min-max."""
        v = torch.cat([x.reshape(-1, x.shape[-1]) for x in self.values], dim=0)
        w = torch.cat([x.reshape(-1) for x in self.weights], dim=0)
        v = v.to(quantizer.scale.device)
        w = w.to(quantizer.scale.device)

        err = (quantizer(v) - v).pow(2)
        return {
            "mse": float(err.mean()),
            "weighted_mse": float((err.mean(dim=-1) * w).sum() / w.sum()),
        }

    def free(self) -> None:
        self.values.clear()
        self.weights.clear()
        self._n_tokens = 0


__all__ = ["InteractionCollector", "interaction_weights"]
