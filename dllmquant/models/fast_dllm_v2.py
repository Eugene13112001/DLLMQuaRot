"""Adapter for Fast-dLLM-v2 (NVIDIA) -- a block diffusion LM built on Qwen2.5-7B.

The third block dLLM in this project, and the one that separates two mechanisms that
LLaDA2.0-mini confounds. It has **no QK-Norm**: its attention projects Q, K and V with a
bias and goes straight to rotary. So whatever fixed-channel structure its keys carry
cannot come from a norm gain, and if the key axis still decides the cache at two bits,
the cause here is the bias in ``k_proj`` -- the mechanism reported for Qwen2.5 elsewhere.
A model where the axis does *not* decide would be just as informative: it would bound the
claim to models that carry a parameter-induced outlier.

What it shares with LLaDA2.0-mini: an exact block-causal prefix, blocks of 32, and a cache
the sampler writes once per block. What it does differently: the prefix cache is a
``transformers`` ``Cache`` the checkpoint's own ``generate`` drives, the current block has
its own separate store (``block_past_key_values``), and decoding commits by confidence
rather than on a schedule. That is the harness of the BitSieve tables, which is why the
numbers taken here can sit next to theirs.

The probe needs nothing new: the projections are ``q_proj``/``k_proj``/``v_proj``, the
pre-attention norm is ``input_layernorm``, rotary covers the whole head, and the mask is
block-causal -- ``q_idx // block >= kv_idx // block``, the checkpoint's ``eval_block_diff_mask``.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import DLLMQuantConfig
from .base import check_transformers_version, load_pretrained
from .llada import LLaDAAdapter

# The sampler's own default; the tokenizer of this checkpoint has no mask_token.
FAST_DLLM_V2_MASK_ID = 151665


class FastDLLMv2Adapter(LLaDAAdapter):
    """Fast-dLLM-v2-7B (and any checkpoint using its vendored modelling code)."""

    # The vendored code imports GradientCheckpointingLayer and
    # ALL_ATTENTION_FUNCTIONS (4.53+) and flex_attention from torch.
    TRANSFORMERS_MIN = (4, 53)
    TRANSFORMERS_MAX = (6, 0)

    def __init__(self, cfg: DLLMQuantConfig):
        super().__init__(cfg)
        self.mask_id = FAST_DLLM_V2_MASK_ID
        self.block_length = cfg.tmas.block_length

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        from transformers import AutoTokenizer

        check_transformers_version(
            minimum=self.TRANSFORMERS_MIN,
            maximum=self.TRANSFORMERS_MAX,
            strict=not getattr(self.cfg, "allow_untested", False),
            model=self.cfg.model_path or "this checkpoint",
        )
        from ..vendor.fast_dllm_v2 import REVISION
        from ..vendor.fast_dllm_v2.configuration import Fast_dLLM_QwenConfig
        from ..vendor.fast_dllm_v2.modeling import Fast_dLLM_QwenForCausalLM

        self.code_revision = REVISION
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_path, trust_remote_code=True
        )
        model_config = Fast_dLLM_QwenConfig.from_pretrained(self.cfg.model_path)
        self.model = load_pretrained(
            Fast_dLLM_QwenForCausalLM, self.cfg,
            self.TRANSFORMERS_MIN, self.TRANSFORMERS_MAX,
            model_config=model_config,
        )
        self.model.eval()
        self.mask_id = self._discover_mask_id()
        # The block grid is the checkpoint's own: the mask, the sampler and this
        # adapter must agree on it, so it is read from the config rather than
        # taken from whatever TMASConfig a caller happens to pass.
        self.block_length = int(getattr(self.model.config, "bd_size", self.block_length))
        self._validate()

    def _discover_mask_id(self) -> int:
        """The mask token, from the tokenizer if it names one, else the sampler's default.

        This checkpoint does not set ``mask_token_id`` anywhere; 151665 is the id its own
        ``generate`` hardcodes. Reading it from the tokenizer first means a re-tokenized
        or extended checkpoint is not silently decoded against the wrong id.
        """
        tok = self.tokenizer
        if getattr(tok, "mask_token_id", None) is not None:
            return int(tok.mask_token_id)
        for literal in ("<|mask|>", "<mask>", "[MASK]", "<mdm_mask>"):
            ids = tok.convert_tokens_to_ids(literal)
            if isinstance(ids, int) and ids >= 0:
                return ids
        return FAST_DLLM_V2_MASK_ID

    # ----------------------------------------------------------- block masking

    def block_attention_mask(
        self,
        seq_len: int,
        block_length: int,
        *,
        batch_size: int = 1,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        """Additive block-causal mask, ``[B, 1, T, T]``: 0 where attention may look.

        The checkpoint's ``eval_block_diff_mask`` is ``q_idx // block >= kv_idx // block``
        as a boolean; the probe adds its mask to the scores, so it is turned into
        0 / -inf here. Bidirectional inside a block, visible to every earlier block.
        """
        idx = torch.arange(seq_len, device=device)
        allowed = (idx[:, None] // block_length) >= (idx[None, :] // block_length)
        mask = allowed.to(dtype or torch.float32).log()
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

    def forward_kwargs(self, input_ids: torch.Tensor) -> Dict[str, object]:
        """Nothing: this model builds its own mask from ``block_size`` inside ``forward``.

        Its default ``block_size`` is 32, the same as ``bd_size`` in the config, so a bare
        ``model(x)`` already attends block-causally. Passing a 4-D mask here would be
        ignored on the cached path and would fight the one the model builds on the other.
        """
        return {}

    # ------------------------------------------------------------ probe pieces

    def _probe_rotary_dim(self) -> Optional[int]:
        """Rotary covers the whole head here -- no partial factor in this family."""
        cfg = self.model.config
        factor = getattr(cfg, "partial_rotary_factor", 1.0) or 1.0
        return int(self.head_dim * factor)

    def _probe_attn_mask_fn(self):
        def mask_fn(seq_len: int, device, dtype):
            return self.block_attention_mask(
                seq_len, self.block_length, device=device, dtype=dtype
            )

        return mask_fn

    def _sequence_layout(self, p_len: int, cfg) -> tuple:
        """Blocks on a grid anchored at position 0, as in LLaDA2.0 and as the mask assumes."""
        b = cfg.block_length
        self.block_length = b
        n_blocks = (p_len + cfg.gen_length + b - 1) // b
        first = p_len // b
        bounds = [(i * b, (i + 1) * b) for i in range(first, n_blocks)]
        return n_blocks * b, bounds

    # -------------------------------------------------------------- reporting

    def describe(self) -> str:
        revision = getattr(self, "code_revision", "?")[:7]
        qk = "none" if self._has_qk_norm() is False else "present"
        return (
            f"FastDLLMv2Adapter [code {revision}]: "
            f"{len(self.blocks)} blocks at '{self._blocks_path}', "
            f"heads={self.n_heads}, kv_heads={self.n_kv_heads}, "
            f"head_dim={self.head_dim}, mask_id={self.mask_id}, "
            f"block={self.block_length}, QK-Norm {qk}, "
            f"k_proj bias {'yes' if self._has_key_bias() else 'no'}"
        )

    def _has_qk_norm(self) -> bool:
        from .llada import _K_NORM_NAMES
        from .base import find_submodule

        return find_submodule(self.blocks[0], _K_NORM_NAMES) is not None

    def _has_key_bias(self) -> bool:
        from .llada import _K_NAMES
        from .base import find_submodule

        k = find_submodule(self.blocks[0], _K_NAMES)
        return k is not None and getattr(k, "bias", None) is not None


__all__ = ["FastDLLMv2Adapter", "FAST_DLLM_V2_MASK_ID"]
