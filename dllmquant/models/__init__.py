from ..config import DLLMQuantConfig
from .base import ArchitectureMismatch, ModelAdapter, discover_blocks
from .llada import LLADA_MASK_ID, LLaDAAdapter
from .fast_dllm_v2 import FAST_DLLM_V2_MASK_ID, FastDLLMv2Adapter
from .llada2_moe import ExpertCoverage, LLaDA2MoEAdapter

ADAPTERS = {
    "llada": LLaDAAdapter,
    "llada2_moe": LLaDA2MoEAdapter,
    "fast_dllm_v2": FastDLLMv2Adapter,
}


def build_adapter(cfg: DLLMQuantConfig) -> ModelAdapter:
    if cfg.model_type not in ADAPTERS:
        raise ValueError(
            f"unknown model_type '{cfg.model_type}'; choose from {sorted(ADAPTERS)}"
        )
    return ADAPTERS[cfg.model_type](cfg)


__all__ = [
    "ModelAdapter",
    "LLaDAAdapter",
    "LLaDA2MoEAdapter",
    "FastDLLMv2Adapter",
    "ExpertCoverage",
    "ArchitectureMismatch",
    "discover_blocks",
    "build_adapter",
    "ADAPTERS",
    "LLADA_MASK_ID",
    "FAST_DLLM_V2_MASK_ID",
]
