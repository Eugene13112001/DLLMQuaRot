"""Quantized drop-in replacements for ``nn.Linear``."""

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QuantConfig
from .quantizers import UniformAffineQuantizer


class QuantLinear(nn.Module):
    """``nn.Linear`` with fake-quantized weights and input activations.

    The FP copy of the weight is kept until CGQ (or RTN) writes the quantized
    values back through :meth:`set_weight`, because GPTQ-style solvers need the
    original weight to compute the error being compensated.
    """

    def __init__(
        self,
        linear: nn.Linear,
        w_cfg: QuantConfig,
        a_cfg: QuantConfig,
        name: str = "",
    ):
        super().__init__()
        self.name = name
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.register_buffer("weight", linear.weight.data.clone())
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None

        self.w_cfg = w_cfg
        self.act_quantizer = UniformAffineQuantizer(a_cfg)

        # Set by IA-AQ on the projection that produces the value matrix.  When
        # QKV are fused, ``out_slice`` restricts it to the V columns.
        self.out_quantizer: Optional[nn.Module] = None
        self.out_slice: Optional[tuple[int, int]] = None

        self.act_quant_enabled: bool = a_cfg.n_bits < 16
        self.weight_quantized: bool = False

        # Calibration plumbing -------------------------------------------------
        self._collecting: bool = False
        self._collected: List[torch.Tensor] = []
        # Streaming alternative to _collected: the CGQ driver installs a
        # callback so Hessians accumulate without holding every activation.
        self._input_callback = None

    # --------------------------------------------------------------- plumbing

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, w_cfg: QuantConfig, a_cfg: QuantConfig, name: str = ""
    ) -> "QuantLinear":
        q = cls(linear, w_cfg, a_cfg, name)
        return q.to(linear.weight.device).to(linear.weight.dtype)

    def set_weight(self, weight: torch.Tensor) -> None:
        """Install the solver's output; marks the layer as weight-quantized."""
        if weight.shape != self.weight.shape:
            raise ValueError(
                f"{self.name}: expected {tuple(self.weight.shape)}, "
                f"got {tuple(weight.shape)}"
            )
        self.weight.data = weight.to(self.weight.dtype).to(self.weight.device)
        self.weight_quantized = True

    @contextmanager
    def collect_inputs(self):
        """Record every input tensor seen inside the block (post act-quant)."""
        self._collecting = True
        self._collected = []
        try:
            yield self._collected
        finally:
            self._collecting = False
            self._collected = []

    @contextmanager
    def act_quant_disabled(self):
        prev = self.act_quant_enabled
        self.act_quant_enabled = False
        try:
            yield
        finally:
            self.act_quant_enabled = prev

    # ---------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_quant_enabled:
            x = self.act_quantizer(x)
        if self._collecting:
            self._collected.append(x.detach())
        if self._input_callback is not None:
            self._input_callback(x.detach())
        out = F.linear(x, self.weight, self.bias)

        if self.out_quantizer is not None:
            if self.out_slice is None:
                out = self.out_quantizer(out)
            else:
                lo, hi = self.out_slice
                head, v, tail = out[..., :lo], out[..., lo:hi], out[..., hi:]
                out = torch.cat([head, self.out_quantizer(v), tail], dim=-1)
        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"w{self.w_cfg.n_bits}a{self.act_quantizer.n_bits}, "
            f"wq={self.weight_quantized}"
        )


def _iter_named_linears(module: nn.Module, prefix: str = ""):
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            yield full, name, module, child
        else:
            yield from _iter_named_linears(child, full)


def wrap_linears(
    block: nn.Module,
    w_cfg: QuantConfig,
    a_cfg: QuantConfig,
    skip: Optional[callable] = None,
    prefix: str = "",
) -> dict[str, QuantLinear]:
    """Replace every ``nn.Linear`` inside ``block`` with a :class:`QuantLinear`.

    Returns a mapping from qualified name to the new module.  ``skip`` receives
    the qualified name and returns True for layers that must stay FP.
    """
    replaced: dict[str, QuantLinear] = {}
    # The list has to be materialised -- the tree is mutated as we go, and
    # walking a module tree while replacing its children is not safe.  But
    # holding it whole keeps every original weight alive until the loop ends,
    # which doubles the block's footprint at the peak: for one LLaDA2.0 MoE
    # block that is 256 experts x 3 projections, about 1.6 GB of bf16 that has
    # already been superseded.  Each entry is therefore dropped as soon as its
    # replacement is installed, so the peak is the block plus one weight.
    targets = list(_iter_named_linears(block, prefix))
    for i, (full, attr, parent, linear) in enumerate(targets):
        if skip is not None and skip(full):
            targets[i] = None
            continue
        q = QuantLinear.from_linear(linear, w_cfg, a_cfg, name=full)
        setattr(parent, attr, q)
        replaced[full] = q
        targets[i] = None
        del linear, parent
    return replaced


def find_quant_linears(module: nn.Module) -> dict[str, QuantLinear]:
    return {
        name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)
    }


__all__ = ["QuantLinear", "wrap_linears", "find_quant_linears"]
