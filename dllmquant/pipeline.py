"""End-to-end DLLMQuant pipeline.

    TMAS  ->  capture block-0 inputs  ->  per block { IA-AQ, CGQ }  ->  save

Blocks are processed one at a time and their inputs are re-generated from the
already-quantized predecessor, so error does not compound the way it would if
every layer were calibrated against FP activations.  That matters more for a
DLLM than for an AR model: the same weights are re-applied at every denoising
step, so a per-layer bias is paid dozens of times per sample.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .algos.cgq import CGQ, certainty_weights
from .algos.ia_aq import InteractionCollector
from .calib.tmas import Snapshot, build_calibration_set, text_calibration_set
from .checkpoint import BlockCheckpoints, config_fingerprint
from .config import DLLMQuantConfig
from .models.base import ModelAdapter
from .modules import QuantLinear, find_quant_linears, wrap_linears

# Linears are quantized in these groups, each group calibrated against the
# already-quantized output of the previous one.
SEQUENTIAL_GROUPS: List[List[str]] = [
    ["att_proj", "qkv_proj", "Wqkv", "q_proj", "k_proj", "v_proj"],
    ["attn_out", "o_proj", "out_proj"],
    ["ff_proj", "up_proj", "gate_proj", "w1", "w3"],
    ["ff_out", "down_proj", "w2"],
]


class _Stop(Exception):
    pass


class _Catcher(nn.Module):
    """Intercepts the inputs of block 0 and aborts the forward pass."""

    def __init__(self, block: nn.Module, store: dict):
        super().__init__()
        self.block = block
        self.store = store

    def forward(self, *args, **kwargs):
        hidden = kwargs.pop("hidden_states", None)
        if hidden is None and args:
            hidden, args = args[0], args[1:]
        self.store["inps"].append(hidden.detach())
        self.store["kwargs"].append(kwargs)
        raise _Stop


@dataclass
class LayerReport:
    name: str
    block: int
    proxy_loss: float
    tokens: float
    weighted: bool
    seconds: float


@dataclass
class QuantReport:
    layers: List[LayerReport] = field(default_factory=list)
    ia_aq: Dict[str, Dict[str, float]] = field(default_factory=dict)
    n_calibration: int = 0
    unweighted_layers: List[str] = field(default_factory=list)
    rotation: Optional[object] = None  # RotationReport, if --rotate was used
    restored_blocks: set = field(default_factory=set)
    seconds: float = 0.0

    def summary(self) -> str:
        if not self.layers:
            return "no layers quantized"
        losses = [l.proxy_loss for l in self.layers]
        out = []
        if self.rotation is not None:
            out.append(self.rotation.summary())
        out += [
            f"quantized {len(self.layers)} layers in {self.seconds:.1f}s",
            f"  calibration snapshots: {self.n_calibration}",
            f"  proxy loss  mean {sum(losses) / len(losses):.5f}  "
            f"max {max(losses):.5f} ({max(self.layers, key=lambda l: l.proxy_loss).name})",
        ]
        if self.ia_aq:
            gains = [
                v["minmax_weighted_mse"] / max(v["weighted_mse"], 1e-12)
                for v in self.ia_aq.values()
                if "minmax_weighted_mse" in v
            ]
            if gains:
                out.append(
                    f"  IA-AQ weighted-MSE improvement: "
                    f"{sum(gains) / len(gains):.2f}x mean over {len(gains)} layers"
                )
        if self.restored_blocks:
            out.append(
                f"  {len(self.restored_blocks)} blocks restored from checkpoint "
                f"(not re-solved)"
            )
        if self.unweighted_layers:
            out.append(
                f"  !! {len(self.unweighted_layers)} layers fell back to uniform "
                f"token weights (token count did not match the snapshot; "
                f"expected for MoE experts): e.g. {self.unweighted_layers[:3]}"
            )
        return "\n".join(out)


class DLLMQuantPipeline:
    def __init__(self, cfg: DLLMQuantConfig, adapter: ModelAdapter):
        self.cfg = cfg
        self.adapter = adapter
        self.report = QuantReport()

    # ------------------------------------------------------------------- run

    @torch.no_grad()
    def run(self, prompts: Sequence[torch.Tensor], verbose: bool = True) -> QuantReport:
        t0 = time.time()
        torch.manual_seed(self.cfg.seed)

        if verbose:
            print(self.adapter.describe())

        # 0. QuaRot --------------------------------------------------------
        # Strictly before calibration: CGQ's Hessians and IA-AQ's value scales
        # are only meaningful in the basis the model will actually run in.
        if self.cfg.rotation.enabled:
            from .rotate import apply_quarot

            self.report.rotation = apply_quarot(self.adapter, self.cfg)
            if verbose:
                print(self.report.rotation.summary())

        ckpt = None
        if self.cfg.checkpoint_dir:
            ckpt = BlockCheckpoints(
                self.cfg.checkpoint_dir, config_fingerprint(self.cfg)
            )

        # 1. Calibration set -------------------------------------------------
        if ckpt is not None and ckpt.load_snapshots() is not None:
            snapshots = ckpt.load_snapshots()
            if verbose:
                print(f"[ckpt] reusing {len(snapshots)} calibration snapshots")
        elif self.cfg.tmas.mode == "text":
            # Baseline path: no trajectory at all. Rolling out generations is
            # by far the most expensive stage, so this is also much faster.
            snapshots = text_calibration_set(prompts, self.cfg.tmas, verbose=verbose)
        else:
            snapshots = build_calibration_set(
                lambda p: self.adapter.trace_generation(p, self.cfg.tmas),
                prompts,
                self.cfg.tmas,
                verbose=verbose,
            )
        self.report.n_calibration = len(snapshots)
        if ckpt is not None and ckpt.load_snapshots() is None:
            ckpt.save_snapshots(snapshots)

        # 2. Block-0 inputs -------------------------------------------------
        inps, kwargs_list = self._capture_block_inputs(snapshots, verbose)

        # 3. Block by block -------------------------------------------------
        blocks = self.adapter.blocks
        for bi, block in enumerate(blocks):
            t_block = time.time()
            layers = wrap_linears(
                block,
                self.cfg.weight,
                self.cfg.activation,
                skip=self.cfg.skip,
                prefix=f"blocks.{bi}",
            )
            if not layers:
                inps = self._forward_block(block, inps, kwargs_list)
                continue

            if ckpt is not None and ckpt.has_block(bi):
                self._restore_block(ckpt, bi, layers)
                inps = self._forward_block(block, inps, kwargs_list)
                if verbose:
                    print(f"[ckpt] block {bi + 1}/{len(blocks)} restored")
                continue

            if self.cfg.ia_aq.enabled:
                self._run_ia_aq(bi, block, layers, inps, kwargs_list, verbose)

            n_before = len(self.report.layers)
            self._run_cgq(bi, block, layers, inps, kwargs_list, snapshots, verbose)

            if ckpt is not None:
                ckpt.save_block(
                    bi,
                    {n: l.weight for n, l in layers.items()},
                    self._value_quantizer_state(layers),
                    [vars(l) for l in self.report.layers[n_before:]],
                )

            inps = self._forward_block(block, inps, kwargs_list)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if verbose:
                print(f"[block {bi + 1}/{len(blocks)}] "
                      f"{len(layers)} layers, {time.time() - t_block:.1f}s")

        self.report.seconds = time.time() - t0
        return self.report

    # --------------------------------------------------------------- stage 2

    def _capture_block_inputs(
        self, snapshots: Sequence[Snapshot], verbose: bool
    ) -> Tuple[List[torch.Tensor], List[dict]]:
        blocks = self.adapter.blocks
        store = {"inps": [], "kwargs": []}
        blocks[0] = _Catcher(blocks[0], store)

        device = next(self.adapter.model.parameters()).device
        for i, snap in enumerate(snapshots):
            try:
                self.adapter.model(snap.input_ids.unsqueeze(0).to(device))
            except _Stop:
                pass
            if verbose and (i + 1) % 32 == 0:
                print(f"  [capture] {i + 1}/{len(snapshots)}")

        blocks[0] = blocks[0].block
        if not store["inps"]:
            raise RuntimeError(
                "captured no block inputs -- the catcher never fired; the "
                "discovered block list is probably not on the forward path"
            )
        return store["inps"], store["kwargs"]

    # ------------------------------------------------------------ checkpoints

    @staticmethod
    def _value_quantizer_state(layers: Dict[str, QuantLinear]) -> Optional[dict]:
        for name, layer in layers.items():
            q = layer.out_quantizer
            if q is None:
                continue
            return {
                "layer": name,
                "slice": list(layer.out_slice) if layer.out_slice else None,
                "scale": q.scale.detach().cpu(),
                "zero": q.zero_point.detach().cpu(),
                "n_bits": q.n_bits,
                "granularity": q.granularity,
                "symmetric": q.symmetric,
            }
        return None

    def _restore_block(self, ckpt, bi: int, layers: Dict[str, QuantLinear]) -> None:
        from .config import QuantConfig
        from .quantizers import InteractionAwareQuantizer

        data = ckpt.load_block(bi)

        missing = set(layers) - set(data["weights"])
        extra = set(data["weights"]) - set(layers)
        if missing or extra:
            raise RuntimeError(
                f"checkpoint for block {bi} does not match the model: "
                f"missing {sorted(missing)}, unexpected {sorted(extra)}. "
                "The checkpoint was written for a different architecture."
            )

        for name, layer in layers.items():
            layer.set_weight(data["weights"][name])
            self.report.restored_blocks.add(bi)

        state = data.get("v_quant")
        if state is not None:
            target = layers[state["layer"]]
            q = InteractionAwareQuantizer(
                QuantConfig(
                    n_bits=state["n_bits"],
                    symmetric=state["symmetric"],
                    granularity=state["granularity"],
                )
            )
            device = target.weight.device
            q.scale = state["scale"].to(device)
            q.zero_point = state["zero"].to(device)
            q.calibrated = True
            q.freeze()
            target.out_quantizer = q
            target.out_slice = tuple(state["slice"]) if state["slice"] else None

        for entry in data.get("layer_reports", []):
            self.report.layers.append(LayerReport(**entry))

    def _forward_block(
        self, block: nn.Module, inps: List[torch.Tensor], kwargs_list: List[dict]
    ) -> List[torch.Tensor]:
        outs = []
        for x, kw in zip(inps, kwargs_list):
            out = block(x, **kw)
            outs.append((out[0] if isinstance(out, tuple) else out).detach())
        return outs

    # --------------------------------------------------------------- stage 3a

    def _run_ia_aq(
        self,
        bi: int,
        block: nn.Module,
        layers: Dict[str, QuantLinear],
        inps: List[torch.Tensor],
        kwargs_list: List[dict],
        verbose: bool,
    ) -> None:
        """Calibrate the value-matrix quantizer from real attention weights."""
        probe = self.adapter.make_probe(block)
        collector = InteractionCollector(self.cfg.ia_aq)

        with probe:
            for x, kw in zip(inps, kwargs_list):
                block(x, **kw)
                if probe.parts is not None:
                    collector.add(probe.parts.value_states, probe.parts.attn_probs)

        if not collector.values:
            if verbose:
                print(f"  [block {bi}] IA-AQ skipped: probe captured nothing")
            return

        device = next(block.parameters()).device
        quantizer = collector.build_quantizer(device=device)

        stats = collector.error_report(quantizer)
        stats["minmax_weighted_mse"] = self._minmax_baseline(collector, device)
        stats["rope_applied"] = float(getattr(probe, "rope_applied", False))
        self.report.ia_aq[f"block{bi}"] = stats

        self._attach_value_quantizer(layers, quantizer, verbose, bi)
        collector.free()

    def _minmax_baseline(self, collector: InteractionCollector, device) -> float:
        """Weighted MSE of a plain min/max scale, for the ablation column."""
        from .config import QuantConfig
        from .quantizers import UniformAffineQuantizer

        v = torch.cat([x.reshape(-1, x.shape[-1]) for x in collector.values]).to(device)
        w = torch.cat([x.reshape(-1) for x in collector.weights]).to(device)
        q = UniformAffineQuantizer(
            QuantConfig(
                n_bits=self.cfg.ia_aq.n_bits,
                granularity="per_feature",
                mse_search=False,
            )
        ).to(device)
        q.find_params(v)
        err = (q(v) - v).pow(2)
        return float((err.mean(dim=-1) * w).sum() / w.sum())

    def _attach_value_quantizer(
        self,
        layers: Dict[str, QuantLinear],
        quantizer: nn.Module,
        verbose: bool,
        bi: int,
    ) -> None:
        d_q = self.adapter.n_heads * self.adapter.head_dim
        d_kv = self.adapter.n_kv_heads * self.adapter.head_dim

        for name, layer in layers.items():
            leaf = name.split(".")[-1]
            if leaf in ("v_proj", "wv", "value"):
                layer.out_quantizer = quantizer
                layer.out_slice = None
                return
            if leaf in ("att_proj", "qkv_proj", "Wqkv"):
                layer.out_quantizer = quantizer
                layer.out_slice = (d_q + d_kv, d_q + 2 * d_kv)
                return

        if verbose:
            print(f"  [block {bi}] IA-AQ: no value projection found among "
                  f"{list(layers)[:6]}; quantizer not attached")

    # --------------------------------------------------------------- stage 3b

    def _run_cgq(
        self,
        bi: int,
        block: nn.Module,
        layers: Dict[str, QuantLinear],
        inps: List[torch.Tensor],
        kwargs_list: List[dict],
        snapshots: Sequence[Snapshot],
        verbose: bool,
    ) -> None:
        remaining = dict(layers)

        for group in SEQUENTIAL_GROUPS + [None]:
            if group is None:
                selected = dict(remaining)  # whatever patterns did not match
            else:
                selected = {
                    n: l for n, l in remaining.items()
                    if n.split(".")[-1] in group
                }
            if not selected:
                continue
            for n in selected:
                remaining.pop(n, None)

            self._solve_group(bi, block, selected, inps, kwargs_list, snapshots, verbose)

    def _solve_group(
        self,
        bi: int,
        block: nn.Module,
        layers: Dict[str, QuantLinear],
        inps: List[torch.Tensor],
        kwargs_list: List[dict],
        snapshots: Sequence[Snapshot],
        verbose: bool,
    ) -> None:
        device = next(block.parameters()).device
        solvers = {n: CGQ(l, self.cfg.cgq, self.cfg.weight) for n, l in layers.items()}
        state = {"weights": None, "expected": 0}

        def make_cb(name: str):
            def cb(x: torch.Tensor):
                n_tokens = x.numel() // x.shape[-1]
                w = state["weights"]
                if w is None or w.numel() != n_tokens:
                    if name not in self.report.unweighted_layers and w is not None:
                        self.report.unweighted_layers.append(name)
                    solvers[name].add_batch(x, None)
                else:
                    solvers[name].add_batch(x, w)
            return cb

        for name, layer in layers.items():
            layer._input_callback = make_cb(name)

        try:
            for x, kw, snap in zip(inps, kwargs_list, snapshots):
                state["weights"] = certainty_weights(
                    snap.mask, snap.confidence, self.cfg.cgq
                ).to(device)
                block(x, **kw)
        finally:
            for layer in layers.values():
                layer._input_callback = None

        for name, layer in layers.items():
            t0 = time.time()
            q, loss = solvers[name].quantize()
            layer.set_weight(q)
            self.report.layers.append(
                LayerReport(
                    name=name,
                    block=bi,
                    proxy_loss=loss,
                    tokens=solvers[name].total_weight,
                    weighted=name not in self.report.unweighted_layers,
                    seconds=time.time() - t0,
                )
            )
            solvers[name].free()
            if verbose:
                print(f"    {name}: proxy loss {loss:.5f} "
                      f"({time.time() - t0:.1f}s)")


__all__ = ["DLLMQuantPipeline", "QuantReport", "LayerReport", "SEQUENTIAL_GROUPS"]
