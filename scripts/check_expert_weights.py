"""Does CGQ's certainty weighting actually reach the MoE experts?

It did not. A 20-hour W4A4 run reported 14316 of 14692 layers falling back to
uniform token weights -- and 14692 minus 14316 is exactly the number of layers
that saw no calibration token at all, so *every* expert layer that saw a token
fell back. Certainty weighting is the diffusion-specific half of CGQ; without
it the run produced QuaRot + plain GPTQ under a CGQ label.

Finding out which of four things went wrong should not cost another 20 hours,
and it does not have to: the question is entirely about one block's plumbing.
Does the router hook fire during a block forward, and do the row counts
reconcile? Neither needs a trajectory, a solver, or the other 19 blocks.

    bash scripts/llada2.sh scripts/check_expert_weights.py \
        --model inclusionAI/LLaDA2.0-mini --model-type llada2_moe --device cpu

Prints, per expert, what the layer saw against what the recorded routing says
it should have seen -- and the reason string the pipeline would have recorded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dllmquant.algos.cgq import certainty_weights  # noqa: E402
from dllmquant.config import CGQConfig, DLLMQuantConfig, ROUTER_NAMES  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402
from dllmquant.models.llada2_local import block_causal_mask  # noqa: E402
from dllmquant.pipeline import _expert_index, _weights_for_expert  # noqa: E402


def _rotary(model):
    """The model's own rotary module, wherever it hangs.

    Found rather than hardcoded: `model.model.rotary_emb` holds for this
    checkpoint's causal-LM wrapper and would break silently on a bare base
    model, and this checkpoint uses partial RoPE over 64 of 128 head
    dimensions, so rebuilding one here would be a second implementation of the
    thing most likely to be got wrong.
    """
    for path in (("model", "rotary_emb"), ("rotary_emb",)):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    for name, module in model.named_modules():
        if name.split(".")[-1] == "rotary_emb":
            return module
    raise AttributeError("no rotary_emb found; a decoder layer cannot be "
                         "called on its own without one")


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
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--block", type=int, default=1,
                    help="which decoder block to probe; 0 is dense in this "
                         "checkpoint (first_k_dense_replace), so 1 is the "
                         "first one with experts")
    ap.add_argument("--show", type=int, default=8, help="experts to list")
    args = ap.parse_args()

    cfg = DLLMQuantConfig(model_path=args.model, model_type=args.model_type,
                          dtype=args.dtype, device=args.device,
                          device_map=args.device_map)
    adapter = build_adapter(cfg)
    adapter.load()
    print(adapter.describe())

    model = adapter.model
    block = adapter.blocks[args.block]
    device = next(block.parameters()).device
    dtype = next(block.parameters()).dtype
    d_model = int(model.config.hidden_size)
    seq = args.seq_len

    # A block input and a calibration snapshot of the shape the pipeline would
    # hand it. The values do not matter -- only the bookkeeping does.
    torch.manual_seed(0)
    x = torch.randn(1, seq, d_model, device=device, dtype=dtype) * 0.02
    mask = torch.zeros(1, seq, dtype=torch.bool, device=device)
    mask[:, seq // 2:] = True
    confidence = torch.rand(1, seq, device=device) * mask
    weights = certainty_weights(mask, confidence, CGQConfig()).to(device)

    # A decoder layer takes the 4-D mask directly; going through the adapter's
    # forward_kwargs would build the same thing from token ids we do not have.
    #
    # Rotary is computed once at model level and handed down as a ready (cos,
    # sin) pair, so a layer called on its own gets `position_embeddings=None`
    # and unpacks it into a TypeError. Borrow the model's own rotary module
    # rather than rebuilding one: this checkpoint uses partial RoPE over 64 of
    # 128 head dimensions, and a second implementation of that is a second
    # thing to get wrong.
    position_ids = torch.arange(seq, device=device).unsqueeze(0)
    kw = {
        "attention_mask": block_causal_mask(
            seq, args.block_length, batch_size=1, device=device, dtype=dtype),
        "position_ids": position_ids,
        "position_embeddings": _rotary(model)(x, position_ids),
    }

    print(f"\nblock {args.block}, {seq} tokens, calibration weights: "
          f"{weights.numel()} values")

    # --- exactly what _solve_group installs -----------------------------
    state: dict = {"routes": None}
    handles = []

    def router_hook(module, inputs, output):
        parts = output if isinstance(output, (tuple, list)) else (output,)
        for t in parts:
            if isinstance(t, torch.Tensor) and t.numel() and not t.is_floating_point():
                state["routes"] = t.detach()
                return

    routers = []
    for name, module in block.named_modules():
        if name.split(".")[-1] in ROUTER_NAMES and hasattr(module, "num_experts"):
            routers.append(name)
            handles.append(module.register_forward_hook(router_hook))
    print(f"routers hooked: {routers or 'NONE -- this alone explains the fallback'}")

    seen: dict = {}

    def expert_hook(name):
        def hook(module, inputs, output):
            t = inputs[0]
            seen[name] = t.numel() // t.shape[-1]
        return hook

    targets = [n for n, mod in block.named_modules()
               if _expert_index(n) is not None and isinstance(mod, torch.nn.Linear)]
    for name in targets:
        handles.append(dict(block.named_modules())[name]
                       .register_forward_hook(expert_hook(name)))
    print(f"expert linears watched: {len(targets)}")

    try:
        with torch.no_grad():
            block(x, **kw)
    finally:
        for h in handles:
            h.remove()

    routes = state["routes"]
    print(f"\nrouter output captured: "
          f"{tuple(routes.shape) if routes is not None else 'NOTHING'}"
          f"{'' if routes is None else f', dtype {routes.dtype}'}")
    if routes is not None and routes.dim() >= 2:
        print(f"  tokens in routing: {routes.shape[0]}, "
              f"top-k: {routes.shape[-1]}, "
              f"calibration weights: {weights.numel()}")
        if routes.shape[0] != weights.numel():
            print("  ^^ these must be equal; a token in the routing has no "
                  "weight to gather")

    print(f"\n{'layer':<44} {'rows seen':>10}  {'verdict'}")
    ok = 0
    reasons: dict = {}
    for i, name in enumerate(sorted(seen, key=lambda n: (_expert_index(n) or 0, n))):
        n_rows = seen[name]
        w, reason = _weights_for_expert(name, weights, routes, n_rows)
        ok += w is not None
        reasons[reason] = reasons.get(reason, 0) + 1
        if i < args.show:
            print(f"{name:<44} {n_rows:>10}  {reason or 'matched'}")

    print(f"\n{ok} of {len(seen)} expert layers reconciled")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>6} x {reason or 'matched'}")
    if ok == len(seen) and seen:
        print("\nThe plumbing is sound here, so the failure is upstream: the "
              "weights or the routing seen during a real calibration differ "
              "from these. Run quantize.py again -- its report now names the "
              "reason instead of guessing at it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
