"""Where does the windowed forward start to disagree with the full one?

At 16 bits the cache stores the tensor untouched, so a windowed forward and a
full one are the same computation and should differ only by the order the
arithmetic happens in. On this checkpoint they differ enough to move committed
decisions, and a number measured on the logits -- twenty blocks downstream of
wherever it started -- cannot say why.

Two explanations, one benign and one not:

* bfloat16 noise compounding with depth. The windowed pass multiplies a
  32-row query against the keys where the full one multiplies every row, and a
  different reduction shape is a different rounding. Then the divergence
  should start near zero and grow smoothly.
* the router. A MoE picks its experts by a top-k over 256 candidates, and
  `check_router` measured that choice flipping for 5% of tokens under an 8-bit
  perturbation. A route that flips does not add noise -- it swaps the matrices
  the token is multiplied by. Then the divergence should sit at the floor and
  jump at the layer where a route first changes.

The distinction decides whether cache reuse in a MoE diffusion LM is benign at
any bit width, so it is worth measuring rather than assuming.

    bash scripts/llada2.sh scripts/trace_window_path.py \
        --model inclusionAI/LLaDA2.0-mini --model-type llada2_moe --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dllmquant.cache import BlockKVCache, KVCacheConfig  # noqa: E402
from dllmquant.config import DLLMQuantConfig, ROUTER_NAMES  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402
from dllmquant.models.llada2_local import (  # noqa: E402
    install_block_cache,
    logits_for_window,
    refresh_prefix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_block_cache import full_logits, masked_canvas, text_ids  # noqa: E402


class Trace:
    """Per-layer hidden states and routing choices for one forward."""

    def __init__(self, blocks):
        self.hidden: dict = {}
        self.routes: dict = {}
        self._handles = []

        for i, block in enumerate(blocks):
            self._handles.append(
                block.register_forward_hook(self._hidden_hook(i))
            )
            for name, module in block.named_modules():
                if (name.split(".")[-1] in ROUTER_NAMES
                        and hasattr(module, "num_experts")):
                    self._handles.append(
                        module.register_forward_hook(self._route_hook(i))
                    )

    def _hidden_hook(self, i):
        def hook(module, inputs, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(t, torch.Tensor):
                self.hidden[i] = t.detach().float()
        return hook

    def _route_hook(self, i):
        def hook(module, inputs, output):
            parts = output if isinstance(output, (tuple, list)) else (output,)
            for t in parts:
                if isinstance(t, torch.Tensor) and t.numel() and not t.is_floating_point():
                    self.routes[i] = t.detach()
                    return
        return hook

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def route_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Share of each token's expert set that survives, compared as a set.

    As a set and not positionally: top-k returns its picks ordered by score,
    so two runs that chose the same eight experts in a different order agree
    completely and a positional comparison would call that a change. That
    mistake was made once already in this project and it inverted a result.
    """
    if a.shape != b.shape:
        return float("nan")
    kept = 0
    for row_a, row_b in zip(a.tolist(), b.tolist()):
        kept += len(set(row_a) & set(row_b)) / max(len(row_a), 1)
    return kept / max(a.shape[0], 1)


def force_routes(blocks, routes: dict, lo: int, hi: int):
    """Make every router pick what the exact run picked. Returns an undo.

    The trace can only show that error and route disagreement rise together;
    which one drives the other it cannot say. Pinning the routes settles it.
    If the divergence then stays at the bfloat16 floor instead of growing
    sixty-fold, the router is the amplifier and not a bystander.

    Patched at `group_limited_topk` rather than at the gate's forward, so the
    gathering of scores at the chosen indices, the normalisation and the
    routed_scaling_factor all still run in the checkpoint's own code. Only the
    choice is overridden -- reimplementing the rest here would be a second
    implementation of the exact thing under test.
    """
    undo = []
    for i, block in enumerate(blocks):
        if i not in routes:
            continue
        pinned = routes[i]
        if pinned.shape[0] > hi - lo:
            pinned = pinned[lo:hi]
        for name, module in block.named_modules():
            if (name.split(".")[-1] in ROUTER_NAMES
                    and hasattr(module, "group_limited_topk")):
                original = module.group_limited_topk

                def patched(scores, _pinned=pinned, _orig=original):
                    idx = _pinned.to(scores.device)
                    if idx.shape[0] != scores.shape[0]:
                        return _orig(scores)          # shape moved; do not guess
                    # The first return value is discarded by the caller.
                    return torch.gather(scores, 1, idx), idx

                module.group_limited_topk = patched
                undo.append((module, original))

    def restore():
        for module, original in undo:
            module.group_limited_topk = original

    return restore


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="llada2_moe")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--device-map", default=None,
                    help="spread the model over several visible cards, e.g. "
                         "'auto'. Use when no single card holds it: with "
                         "CUDA_VISIBLE_DEVICES=3,1 the layers are split and "
                         "the cost is one activation crossing per boundary, "
                         "about 1 MB -- far cheaper than sharing one busy "
                         "card, where neighbours cost a factor of thirty.")
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--prefix-mask-ratio", type=float, default=0.0)
    ap.add_argument("--no-pin", action="store_true",
                    help="skip the pinned-routes run (one extra windowed "
                         "forward); leaves only the correlation")
    args = ap.parse_args()

    cfg = DLLMQuantConfig(model_path=args.model, model_type=args.model_type,
                          dtype=args.dtype, device=args.device,
                          device_map=args.device_map)
    adapter = build_adapter(cfg)
    adapter.load()
    print(adapter.describe())

    model = adapter.model
    device = next(model.parameters()).device
    n_layers = len(adapter.blocks)
    total = args.blocks * args.block_length
    prefix_len = total - args.block_length

    x = masked_canvas(adapter, text_ids(adapter, total), args.block_length,
                      prefix_len, args.prefix_mask_ratio, 1.0).to(device)

    # --- the exact computation ------------------------------------------
    full = Trace(adapter.blocks)
    reference = full_logits(model, x, args.block_length)
    full.close()

    # --- the same thing, prefix read from a lossless cache ---------------
    states = install_block_cache(model, BlockKVCache(KVCacheConfig(), n_layers))
    cache = BlockKVCache(
        KVCacheConfig(enabled=True, decoded_bits=16, masked_bits=16,
                      group_size=adapter.head_dim),
        n_layers,
    )
    for state in states:
        state.cache = cache
    refresh_prefix(model, states, x, prefix_len, args.block_length)

    windowed_trace = Trace(adapter.blocks)
    windowed = logits_for_window(model, states, x, prefix_len, total,
                                 args.block_length)
    windowed_trace.close()

    # --- and again with the expert choice pinned to the exact run's ------
    pinned_trace = pinned_logits = None
    if not args.no_pin:
        restore = force_routes(adapter.blocks, full.routes, prefix_len, total)
        try:
            refresh_prefix(model, states, x, prefix_len, args.block_length)
            pinned_trace = Trace(adapter.blocks)
            pinned_logits = logits_for_window(model, states, x, prefix_len,
                                              total, args.block_length)
            pinned_trace.close()
        finally:
            restore()

    print(f"\n{total} tokens, prefix {prefix_len} cached at 16 bits (a no-op), "
          f"window {args.block_length} recomputed")
    print("both runs compute the same thing; every difference below is the "
          "order of the arithmetic, or something the arithmetic set off\n")

    pin_col = "" if pinned_trace is None else f" {'pinned err':>12}"
    print(f"{'layer':>5} {'hidden rel. err':>16} {'routes kept':>12}{pin_col}")
    prev = 0.0
    jumped = None
    for i in range(n_layers):
        a = full.hidden.get(i)
        b = windowed_trace.hidden.get(i)
        if a is None or b is None:
            continue
        a_win = a[:, prefix_len:total] if a.shape[1] == total else a

        def err(t):
            return float((t - a_win).abs().mean()
                         / a_win.abs().mean().clamp(min=1e-8))

        rel = err(b)
        pin_txt = ""
        if pinned_trace is not None and i in pinned_trace.hidden:
            pin_txt = f" {err(pinned_trace.hidden[i]):>12.3e}"

        ra, rb = full.routes.get(i), windowed_trace.routes.get(i)
        kept = float("nan")
        if ra is not None and rb is not None:
            n_tok = ra.shape[0]
            ra_win = ra[prefix_len:total] if n_tok == total else ra
            kept = route_overlap(ra_win, rb)

        flag = ""
        if jumped is None and prev > 0 and rel > 4 * prev:
            jumped, flag = i, "   <-- jumps here"
        prev = max(rel, 1e-12)
        kept_txt = "     n/a" if kept != kept else f"{100 * kept:7.2f}%"
        print(f"{i:>5} {rel:>16.3e} {kept_txt:>12}{pin_txt}{flag}")

    ref_win = reference[:, prefix_len:]

    def argmax_kept(logits):
        return float((logits.argmax(-1) == ref_win.argmax(-1)).float().mean())

    agree = argmax_kept(windowed)
    print(f"\nlogits: argmax kept {100 * agree:.2f}%")
    if pinned_logits is not None:
        pinned_agree = argmax_kept(pinned_logits)
        print(f"        argmax kept {100 * pinned_agree:.2f}% with the expert "
              f"choice pinned to the exact run")
        print("\nThe pinned column is the experiment the trace alone cannot "
              "do. Error and route disagreement rise together, and correlation "
              "does not say which drives which; forcing every router to repeat "
              "the exact run's choice removes one of them. If the pinned error "
              "stays near the first layer's -- the bfloat16 floor -- then the "
              "router is what turns rounding into a sixty-fold divergence, and "
              "cache reuse in a MoE is not free at any width. If it still "
              "grows, the routes were a symptom and the arithmetic compounds "
              "on its own.")

    routed = [full.routes.get(i) is not None for i in range(n_layers)]
    if any(routed):
        first_route_change = next(
            (i for i in range(n_layers)
             if full.routes.get(i) is not None
             and windowed_trace.routes.get(i) is not None
             and route_overlap(
                 full.routes[i][prefix_len:total]
                 if full.routes[i].shape[0] == total else full.routes[i],
                 windowed_trace.routes[i]) < 1.0),
            None,
        )
        if first_route_change is None:
            print("\nEvery router made the identical choice in both runs, so "
                  "the divergence is arithmetic alone -- read the hidden "
                  "column: growing smoothly with depth is bfloat16, and a "
                  "jump is a bug at that layer.")
        else:
            print(f"\nThe first router to choose differently is layer "
                  f"{first_route_change}. Below that the two runs differ by "
                  "rounding; from there they are running different experts, "
                  "which is not a perturbation of the same computation but a "
                  "different one. That is the mechanism to report: in a MoE "
                  "the router turns arithmetic noise into a discrete change, "
                  "so cache reuse is not free even at 16 bits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
