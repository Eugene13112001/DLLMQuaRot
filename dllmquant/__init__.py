"""DLLMQuant -- post-training W4A4 quantization for diffusion LLMs.

Reimplementation of "DLLMQuant: Quantizing Diffusion-based Large Language
Models" (arXiv:2508.14090) from the paper text; no official code was released
at the time of writing, so numeric parity with the published tables is not
guaranteed.  See README.md.
"""

from .config import (
    CGQConfig,
    DLLMQuantConfig,
    IAAQConfig,
    QuantConfig,
    RotationConfig,
    TMASConfig,
    w4a4,
    w4a16,
)
from .modules import QuantLinear, find_quant_linears, wrap_linears
from .quantizers import InteractionAwareQuantizer, UniformAffineQuantizer

__version__ = "0.1.0"

__all__ = [
    "QuantConfig",
    "TMASConfig",
    "CGQConfig",
    "IAAQConfig",
    "RotationConfig",
    "DLLMQuantConfig",
    "w4a4",
    "w4a16",
    "UniformAffineQuantizer",
    "InteractionAwareQuantizer",
    "QuantLinear",
    "wrap_linears",
    "find_quant_linears",
]
