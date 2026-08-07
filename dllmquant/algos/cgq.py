"""Certainty-Guided Quantization.

GPTQ builds the Hessian ``H = 2 X X^T`` treating every calibration token as
equally worth compensating.  In a DLLM the sequence is a mixture of two
populations: positions already decoded (frozen for the rest of generation) and
positions still masked (about to be committed, and the ones the next denoising
step actually reads).  CGQ reweights the Hessian by token,

    H = 2 * X diag(c) X^T,
    c_j = unmasked_weight                    if j is decoded
          masked_base + beta * conf_j        if j is masked,

so error compensation is spent where it still changes the output.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from ..config import CGQConfig, QuantConfig
from ..modules import QuantLinear
from ..quantizers import UniformAffineQuantizer


def certainty_weights(
    mask: torch.Tensor,
    confidence: torch.Tensor,
    cfg: CGQConfig,
) -> torch.Tensor:
    """Per-token Hessian weights from mask status and prediction confidence.

    ``mask`` is True where the token is still [MASK]; ``confidence`` is the
    model's max softmax probability at that position (meaningless, and ignored,
    where the token is already decoded).
    """
    conf = confidence.clamp(0.0, 1.0).to(torch.float32)
    masked_w = cfg.masked_base + cfg.beta * conf
    unmasked_w = torch.full_like(conf, cfg.unmasked_weight)
    c = torch.where(mask.bool(), masked_w, unmasked_w)

    if cfg.normalize:
        total = c.sum().clamp(min=1e-8)
        c = c * (c.numel() / total)
    return c


class _RowQuantizer:
    """Weight quantizer holding per-row (and optionally per-group) params."""

    def __init__(self, cfg: QuantConfig, weight: torch.Tensor):
        self.cfg = cfg
        self.rows, self.columns = weight.shape
        self.group_size = cfg.group_size if cfg.granularity == "per_group" else -1
        self._q = UniformAffineQuantizer(cfg)
        self.scale: Optional[torch.Tensor] = None
        self.zero: Optional[torch.Tensor] = None
        self.groups: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def find_params_rows(self, weight: torch.Tensor) -> None:
        """One scale per output row, from the whole row."""
        q = UniformAffineQuantizer(
            QuantConfig(
                n_bits=self.cfg.n_bits,
                symmetric=self.cfg.symmetric,
                granularity="per_channel",
                mse_search=self.cfg.mse_search,
                mse_candidates=self.cfg.mse_candidates,
                mse_min_ratio=self.cfg.mse_min_ratio,
            )
        )
        q.find_params(weight)
        self.scale, self.zero = q.scale, q.zero_point
        self._q = q

    def find_params_group(self, weight_slice: torch.Tensor, group_id: int) -> None:
        q = UniformAffineQuantizer(
            QuantConfig(
                n_bits=self.cfg.n_bits,
                symmetric=self.cfg.symmetric,
                granularity="per_channel",
                mse_search=self.cfg.mse_search,
                mse_candidates=self.cfg.mse_candidates,
                mse_min_ratio=self.cfg.mse_min_ratio,
            )
        )
        q.find_params(weight_slice)
        self.groups[group_id] = (q.scale, q.zero_point)
        self.scale, self.zero = q.scale, q.zero_point
        self._q = q

    def use_group(self, group_id: int) -> None:
        self.scale, self.zero = self.groups[group_id]

    def quantize_column(self, col: torch.Tensor) -> torch.Tensor:
        """``col`` is [rows]; scale/zero are [rows, 1]."""
        s = self.scale.reshape(-1)
        z = self.zero.reshape(-1)
        qmin, qmax = self._q.qmin, self._q.qmax
        q = torch.clamp(torch.round(col / s) + z, qmin, qmax)
        return (q - z) * s


class CGQ:
    """Per-layer solver.  Usage: ``add_batch`` over calibration, then ``quantize``."""

    def __init__(self, layer: QuantLinear, cfg: CGQConfig, w_cfg: QuantConfig):
        self.layer = layer
        self.cfg = cfg
        self.w_cfg = w_cfg

        weight = layer.weight.data
        self.rows, self.columns = weight.shape
        self.device = weight.device
        self.H = torch.zeros(
            (self.columns, self.columns), device=self.device, dtype=torch.float32
        )
        self.total_weight = 0.0
        self.n_batches = 0

    # ------------------------------------------------------------------ input

    def add_batch(
        self, inp: torch.Tensor, token_weights: Optional[torch.Tensor] = None
    ) -> None:
        """Accumulate ``2 * X diag(c) X^T``.

        ``inp`` is [..., in_features]; ``token_weights`` is broadcastable to the
        flattened token axis (``inp.numel() // in_features`` entries).
        """
        x = inp.reshape(-1, self.columns).to(torch.float32)
        n_tokens = x.shape[0]

        if token_weights is not None:
            c = token_weights.reshape(-1).to(torch.float32).clamp(min=0.0)
            if c.numel() != n_tokens:
                raise ValueError(
                    f"{self.layer.name}: {c.numel()} token weights for "
                    f"{n_tokens} tokens"
                )
            x = x * c.sqrt().unsqueeze(1)
            self.total_weight += float(c.sum())
        else:
            self.total_weight += float(n_tokens)

        # X is [tokens, in]; H needs [in, in].
        self.H += 2.0 * (x.t() @ x)
        self.n_batches += 1

    def _normalized_hessian(self) -> torch.Tensor:
        if self.total_weight <= 0:
            raise RuntimeError(f"{self.layer.name}: no calibration data collected")
        return self.H / self.total_weight

    # -------------------------------------------------------------- the solve

    @torch.no_grad()
    def quantize(self) -> tuple[torch.Tensor, float]:
        """Return (quantized weight, mean squared proxy loss)."""
        cfg = self.cfg
        W = self.layer.weight.data.clone().to(torch.float32)
        H = self._normalized_hessian().clone()

        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

        rq = _RowQuantizer(self.w_cfg, W)
        group_size = rq.group_size

        # Static groups must be measured before any permutation, otherwise the
        # group a column belongs to depends on the act-order sort.
        if group_size > 0 and cfg.static_groups:
            for g0 in range(0, self.columns, group_size):
                rq.find_params_group(W[:, g0 : g0 + group_size], g0 // group_size)
        elif group_size <= 0:
            rq.find_params_rows(W)

        perm = invperm = None
        if cfg.act_order:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Q = torch.zeros_like(W)
        total_loss = 0.0

        # Hinv = upper Cholesky factor of H^-1.  A calibration set smaller than
        # in_features leaves H rank-deficient, so back off with growing damping
        # instead of dying on a factorization error.
        idx = torch.arange(self.columns, device=self.device)
        base = torch.mean(torch.diag(H)).clamp(min=1e-8)
        Hinv = None
        damp_scale = cfg.percdamp
        for attempt in range(6):
            Hd = H.clone()
            Hd[idx, idx] += damp_scale * base
            try:
                L = torch.linalg.cholesky(Hd)
                Hinv_full = torch.cholesky_inverse(L)
                Hinv = torch.linalg.cholesky(Hinv_full, upper=True)
                break
            except Exception:
                damp_scale *= 10.0
        if Hinv is None:
            raise RuntimeError(
                f"{self.layer.name}: Hessian not positive definite even at "
                f"percdamp={damp_scale:.3g}; collect more calibration tokens"
            )

        blocksize = cfg.blocksize
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                col = i1 + i
                w = W1[:, i]
                d = Hinv1[i, i]

                if group_size > 0:
                    orig_col = int(perm[col]) if perm is not None else col
                    gid = orig_col // group_size
                    if cfg.static_groups:
                        rq.use_group(gid)
                    elif col % group_size == 0:
                        # Dynamic groups: recalibrate on the *remaining* (already
                        # error-compensated) weights of this group.
                        rq.find_params_group(W[:, col : col + group_size], gid)

                q = rq.quantize_column(w)
                Q1[:, i] = q

                err = (w - q) / d
                total_loss += float(((w - q) ** 2).sum() / (d**2))

                W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err

            Q[:, i1:i2] = Q1
            if i2 < self.columns:
                W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        if invperm is not None:
            Q = Q[:, invperm]

        return Q, total_loss / max(self.rows * self.columns, 1)

    def free(self) -> None:
        self.H = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["CGQ", "certainty_weights"]
