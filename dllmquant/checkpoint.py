"""Per-block checkpoints, so a crash does not cost the whole solve.

What has to be saved, and what can be recomputed:

  * the quantized weights of each finished block -- saved, this is the work
    that took the time;
  * the value quantizer IA-AQ calibrated for that block -- saved, it is a few
    numbers and recomputing it would need the attention statistics again;
  * the calibration set -- saved despite being tiny (a few hundred KB),
    because regenerating it means re-running generation, and argmax over a
    non-deterministic matmul can pick a different token and silently give you
    a *different* calibration set on resume;
  * the rotation -- NOT saved. It is a deterministic function of the seed, so
    re-applying it costs a minute and cannot drift.

Resuming still has to push the calibration set through the already-quantized
blocks to reconstruct the inputs of the next one, but forwards are the cheap
part.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional, Tuple

import torch

from .calib.tmas import Snapshot

FORMAT_VERSION = 1


class BlockCheckpoints:
    def __init__(self, directory: str | pathlib.Path, cfg_fingerprint: str):
        self.dir = pathlib.Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fingerprint = cfg_fingerprint
        self._check_fingerprint()

    # ------------------------------------------------------------------ meta

    @property
    def meta_path(self) -> pathlib.Path:
        return self.dir / "meta.json"

    def _check_fingerprint(self) -> None:
        """Refuse to mix blocks quantized under different settings.

        Resuming a w4a4 run into a w3a16 one would produce a model that is
        neither, and nothing downstream would notice.
        """
        if not self.meta_path.exists():
            self.meta_path.write_text(
                json.dumps({"version": FORMAT_VERSION,
                            "fingerprint": self.fingerprint}, indent=2),
                encoding="utf-8",
            )
            return

        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != self.fingerprint:
            raise RuntimeError(
                f"{self.dir} holds blocks quantized with different settings:\n"
                f"  on disk: {meta.get('fingerprint')}\n"
                f"  now:     {self.fingerprint}\n"
                "Point --checkpoint-dir somewhere else, or delete it."
            )
        if meta.get("version") != FORMAT_VERSION:
            raise RuntimeError(
                f"checkpoint format v{meta.get('version')}, this build writes "
                f"v{FORMAT_VERSION}; delete {self.dir} and start over"
            )

    # ------------------------------------------------------- calibration set

    @property
    def snapshots_path(self) -> pathlib.Path:
        return self.dir / "snapshots.pt"

    def save_snapshots(self, snapshots: List[Snapshot]) -> None:
        torch.save(
            [
                {
                    "input_ids": s.input_ids, "mask": s.mask,
                    "confidence": s.confidence, "step": s.step,
                    "total_steps": s.total_steps, "block_idx": s.block_idx,
                    "mask_ratio": s.mask_ratio,
                }
                for s in snapshots
            ],
            self.snapshots_path,
        )

    def load_snapshots(self) -> Optional[List[Snapshot]]:
        if not self.snapshots_path.exists():
            return None
        raw = torch.load(self.snapshots_path, map_location="cpu",
                         weights_only=False)
        return [Snapshot(**d) for d in raw]

    # -------------------------------------------------------------- blocks

    def block_path(self, index: int) -> pathlib.Path:
        return self.dir / f"block_{index:04d}.pt"

    def has_block(self, index: int) -> bool:
        return self.block_path(index).exists()

    def first_unfinished(self, n_blocks: int) -> int:
        """Stop at the first gap: a later block is useless without its
        predecessors, since each one is calibrated on the previous output."""
        for i in range(n_blocks):
            if not self.has_block(i):
                return i
        return n_blocks

    def save_block(
        self,
        index: int,
        weights: Dict[str, torch.Tensor],
        v_quant: Optional[dict],
        layer_reports: List[dict],
    ) -> None:
        tmp = self.block_path(index).with_suffix(".tmp")
        torch.save(
            {
                "weights": {k: v.detach().cpu() for k, v in weights.items()},
                "v_quant": v_quant,
                "layer_reports": layer_reports,
            },
            tmp,
        )
        # Rename last: a half-written file must never look like a finished
        # block, or resume would load garbage into the model.
        tmp.replace(self.block_path(index))

    def load_block(self, index: int) -> dict:
        return torch.load(self.block_path(index), map_location="cpu",
                          weights_only=False)


def config_fingerprint(cfg) -> str:
    """Everything that changes the numbers a block would get."""
    parts = [
        cfg.model_path, cfg.model_type, cfg.dtype,
        f"w{cfg.weight.n_bits}{cfg.weight.granularity}{cfg.weight.group_size}",
        f"sym{int(cfg.weight.symmetric)}mse{int(cfg.weight.mse_search)}",
        f"a{cfg.activation.n_bits}{cfg.activation.granularity}",
        f"clip{cfg.activation.clip_ratio}",
        f"tmas{cfg.tmas.n_samples}:{cfg.tmas.mode}:{int(cfg.tmas.uniform)}"
        f":{cfg.tmas.n_prompts}:{cfg.tmas.steps}:{cfg.tmas.gen_length}",
        f"cgq{cfg.cgq.unmasked_weight}:{cfg.cgq.masked_base}:{cfg.cgq.beta}"
        f":{cfg.cgq.percdamp}:{int(cfg.cgq.act_order)}",
        f"iaaq{int(cfg.ia_aq.enabled)}:{cfg.ia_aq.n_bits}"
        f":{cfg.ia_aq.decoded_query_weight}",
        f"rot{int(cfg.rotation.enabled)}:{int(cfg.rotation.residual)}"
        f":{int(cfg.rotation.value_heads)}:{int(cfg.rotation.online_mlp)}"
        f":{cfg.rotation.seed}",
        f"seed{cfg.seed}",
    ]
    return "|".join(parts)


__all__ = ["BlockCheckpoints", "config_fingerprint", "FORMAT_VERSION"]
