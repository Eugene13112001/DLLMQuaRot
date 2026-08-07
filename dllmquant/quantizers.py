"""Uniform affine fake-quantizers.

Everything here is *simulated* quantization: values are rounded onto a 4-bit
grid and immediately expanded back to the compute dtype.  That measures the
accuracy cost of W4A4 exactly, but does not by itself make anything faster --
real speedups need INT4 kernels (see README, "Что это НЕ делает").
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import QuantConfig


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through gradient (needed only if you fine-tune)."""
    return (x.round() - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    """Uniform affine quantizer with optional (weighted) MSE clip search.

    Shapes follow the convention that the quantized axis is the *last* one:
      * per_channel : one scale per row of a 2-D weight  [out, in]  -> [out, 1]
      * per_token   : one scale per token of an activation [..., d] -> [..., 1]
      * per_group   : one scale per contiguous group of ``group_size`` along -1
      * per_tensor  : a single scalar
    """

    def __init__(self, cfg: QuantConfig):
        super().__init__()
        self.cfg = cfg
        self.n_bits = cfg.n_bits
        self.symmetric = cfg.symmetric
        self.granularity = cfg.granularity
        self.group_size = cfg.group_size

        if cfg.symmetric:
            self.qmin = -(2 ** (cfg.n_bits - 1))
            self.qmax = 2 ** (cfg.n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**cfg.n_bits - 1

        self.register_buffer("scale", torch.zeros(1), persistent=False)
        self.register_buffer("zero_point", torch.zeros(1), persistent=False)
        self.calibrated: bool = False
        self.enabled: bool = cfg.n_bits < 16

    # ------------------------------------------------------------------ utils

    def _reshape_for_group(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """Fold the last dim into groups; returns (view, original shape)."""
        orig = x.shape
        if self.granularity != "per_group":
            return x, orig
        if orig[-1] % self.group_size != 0:
            raise ValueError(
                f"last dim {orig[-1]} not divisible by group_size {self.group_size}"
            )
        return x.reshape(*orig[:-1], orig[-1] // self.group_size, self.group_size), orig

    def _reduce_dims(self, x: torch.Tensor) -> tuple:
        """Dims to reduce over when computing min/max."""
        if self.granularity == "per_tensor":
            return tuple(range(x.dim()))
        if self.granularity == "per_feature":
            # Keep the feature axis, collapse every token axis -- this is what
            # makes the scale shared across tokens (see IA-AQ).
            if x.dim() < 2:
                raise ValueError("per_feature needs at least 2 dims [tokens, features]")
            return tuple(range(x.dim() - 1))
        # per_channel / per_token / per_group keep every leading dim and reduce
        # only the final one.
        return (-1,)

    # ------------------------------------------------------------ calibration

    def find_params(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Compute scale/zero_point from ``x``.

        ``weights`` are optional per-token importances broadcastable to ``x``'s
        leading dims; they only matter when ``cfg.mse_search`` is on, and they
        are what turns this quantizer into the IA-AQ one.
        """
        if not self.enabled:
            self.calibrated = True
            return

        xg, _ = self._reshape_for_group(x)
        dims = self._reduce_dims(xg)

        x_max = xg.amax(dim=dims, keepdim=True)
        x_min = xg.amin(dim=dims, keepdim=True)

        if self.cfg.mse_search:
            scale, zero = self._mse_search(xg, x_min, x_max, weights)
        else:
            r = self.cfg.clip_ratio
            scale, zero = self._params_from_range(x_min * r, x_max * r)

        self.scale = scale
        self.zero_point = zero
        self.calibrated = True

    def _params_from_range(
        self, x_min: torch.Tensor, x_max: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.symmetric:
            amax = torch.maximum(x_max.abs(), x_min.abs())
            scale = amax / max(self.qmax, 1)
            zero = torch.zeros_like(scale)
        else:
            x_min = torch.minimum(x_min, torch.zeros_like(x_min))
            x_max = torch.maximum(x_max, torch.zeros_like(x_max))
            scale = (x_max - x_min) / (self.qmax - self.qmin)
            zero = self.qmin - x_min / scale.clamp(min=1e-8)
            zero = zero.round()
        scale = scale.clamp(min=1e-8)
        return scale, zero

    def _mse_search(
        self,
        x: torch.Tensor,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
        weights: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Grid-search a clipping ratio minimising (weighted) squared error."""
        best_scale, best_zero = self._params_from_range(x_min, x_max)
        best_err = torch.full_like(best_scale, float("inf"))

        w = None
        if weights is not None:
            w = weights.to(x.dtype)
            # Broadcast importance over the quantized axis.
            while w.dim() < x.dim():
                w = w.unsqueeze(-1)

        n = max(self.cfg.mse_candidates, 1)
        lo = self.cfg.mse_min_ratio
        for i in range(n):
            ratio = 1.0 - (1.0 - lo) * i / n
            scale, zero = self._params_from_range(x_min * ratio, x_max * ratio)
            x_hat = self._fake_quant(x, scale, zero)
            err = (x_hat - x).pow_(2)
            if w is not None:
                err = err * w
            err = err.sum(dim=self._reduce_dims(x), keepdim=True)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_scale = torch.where(better, scale, best_scale)
            best_zero = torch.where(better, zero, best_zero)

        return best_scale, best_zero

    # ------------------------------------------------------------------ apply

    def _fake_quant(
        self, x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor
    ) -> torch.Tensor:
        q = torch.clamp(torch.round(x / scale) + zero, self.qmin, self.qmax)
        return (q - zero) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        # Dynamic quantizers take their params from the tensor currently
        # flowing through, not from calibration.  per_token is dynamic by
        # definition; per_group activations must opt in via cfg.dynamic.
        if self.cfg.dynamic or self.granularity == "per_token" or not self.calibrated:
            self.find_params(x)
        xg, orig = self._reshape_for_group(x)
        out = self._fake_quant(xg, self.scale, self.zero_point)
        return out.reshape(orig)

    def extra_repr(self) -> str:
        return (
            f"n_bits={self.n_bits}, {self.granularity}, "
            f"{'sym' if self.symmetric else 'asym'}, enabled={self.enabled}"
        )


class InteractionAwareQuantizer(UniformAffineQuantizer):
    """Static quantizer whose scale was calibrated with attention importances.

    Unlike the per-token dynamic activation quantizer, this one is calibrated
    once (during the IA-AQ pass) and then frozen -- at inference the attention
    matrix is not needed, only the stored scale.
    """

    def __init__(self, cfg: QuantConfig):
        super().__init__(cfg)
        self.frozen = False

    def freeze(self) -> None:
        self.frozen = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if not self.frozen:
            return super().forward(x)
        xg, orig = self._reshape_for_group(x)
        out = self._fake_quant(xg, self.scale, self.zero_point)
        return out.reshape(orig)


def quantize_weight_rtn(weight: torch.Tensor, cfg: QuantConfig) -> torch.Tensor:
    """Round-to-nearest baseline used for sanity checks and ablations."""
    q = UniformAffineQuantizer(cfg)
    q.find_params(weight)
    return q(weight)


__all__ = [
    "UniformAffineQuantizer",
    "InteractionAwareQuantizer",
    "quantize_weight_rtn",
]
