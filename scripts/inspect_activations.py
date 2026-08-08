"""Look at what the activations actually are, before deciding how to quantize.

Three things this answers, none of which you can guess from the architecture:

1. **Which channels are outliers, and how big.** The famous LLM phenomenon: a
   handful of hidden dimensions carry magnitudes 10-100x the rest, and they
   are what defeats per-token activation quantization.

2. **Are they the same channels everywhere?** If one set of channels is
   outlying in every layer, a single fix handles the whole model. If each
   layer has its own, the fix has to be per-layer.

3. **Do they move along the denoising trajectory?** This is the question a
   dense-LLM tool never asks. A diffusion LM sweeps from all-[MASK] to fully
   decoded, and if the outlier set drifts as tokens get committed, then one
   static rotation cannot be equally right at both ends -- which would explain
   why quantization damage concentrates at high mask ratios.

    python scripts/inspect_activations.py --model GSAI-ML/LLaDA-1.5 --out out/acts
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from dllmquant.calib.prompts import load_prompts  # noqa: E402
from dllmquant.config import DLLMQuantConfig, TMASConfig  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402
from dllmquant.report import round_floats, write_csv  # noqa: E402


@torch.no_grad()
def capture_residual_stream(adapter, ids: torch.Tensor) -> Dict[int, torch.Tensor]:
    """Hidden states entering every block, for one input."""
    grabbed: Dict[int, torch.Tensor] = {}
    handles = []

    def make(i):
        def hook(module, args, kwargs, output):
            h = kwargs.get("hidden_states", args[0] if args else None)
            if isinstance(h, torch.Tensor) and h.dim() == 3:
                grabbed[i] = h.detach().float()
        return hook

    for i, block in enumerate(adapter.blocks):
        handles.append(block.register_forward_hook(make(i), with_kwargs=True))
    try:
        adapter.model(ids)
    finally:
        for h in handles:
            h.remove()
    return grabbed


def channel_stats(x: torch.Tensor, top_k: int = 8) -> dict:
    """``x`` is [1, T, D] -- one state's residual stream at one layer."""
    f = x[0]  # [T, D]
    per_channel = f.abs().amax(dim=0)  # [D] worst magnitude each channel reaches
    per_token_rms = f.pow(2).mean(dim=-1).sqrt().clamp(min=1e-8)

    top = torch.topk(per_channel, k=min(top_k, per_channel.numel()))
    median = float(per_channel.median())

    centered = f - f.mean()
    kurt = float((centered.pow(4).mean() / centered.pow(2).mean().pow(2)))

    # Which token carries the single largest value: a "massive activation"
    # usually sits on one delimiter-ish position rather than spreading out.
    flat_idx = int(f.abs().argmax())
    worst_token = flat_idx // f.shape[1]

    return {
        "crest": float((f.abs().amax(dim=-1) / per_token_rms).mean()),
        "kurtosis": kurt,
        "max_abs": float(per_channel.max()),
        "median_channel_abs": median,
        "outlier_ratio": float(per_channel.max()) / max(median, 1e-8),
        "worst_token": worst_token,
        "n_tokens": int(f.shape[0]),
        "top_channels": [int(c) for c in top.indices.tolist()],
        "top_magnitudes": [float(v) for v in top.values.tolist()],
    }


def jaccard(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="llada", choices=["llada", "llada2_moe"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--mask-ratios", type=float, nargs="+",
                    default=[1.0, 0.75, 0.5, 0.25, 0.0])
    ap.add_argument("--out", default="out/acts")
    args = ap.parse_args()

    cfg = DLLMQuantConfig(
        model_path=args.model, model_type=args.model_type,
        dtype=args.dtype, device=args.device,
    )
    adapter = build_adapter(cfg)
    adapter.load()
    print(adapter.describe())
    device = next(adapter.model.parameters()).device

    # Build one input per mask ratio: same prompt, different amounts decoded.
    prompt = adapter.encode_prompts(load_prompts(1, dataset=None), max_len=32)[0]
    filler = adapter.encode_prompts(
        load_prompts(2, dataset=None)[1:], max_len=args.gen_length
    )[0]
    p_len = prompt.shape[-1]

    rows: List[dict] = []
    channel_rows: List[dict] = []
    per_ratio_top: Dict[str, Dict[int, List[int]]] = {}

    for ratio in args.mask_ratios:
        x = torch.full((1, p_len + args.gen_length), adapter.mask_id,
                       dtype=torch.long)
        x[0, :p_len] = prompt
        n_decoded = int(round((1.0 - ratio) * args.gen_length))
        for i in range(n_decoded):
            x[0, p_len + i] = int(filler[i % filler.numel()])

        acts = capture_residual_stream(adapter, x.to(device))
        key = f"{ratio:.2f}"
        per_ratio_top[key] = {}

        for layer, h in sorted(acts.items()):
            s = channel_stats(h, args.top_k)
            per_ratio_top[key][layer] = s["top_channels"]
            rows.append(round_floats({
                "mask_ratio": key, "layer": layer,
                "crest": s["crest"], "kurtosis": s["kurtosis"],
                "max_abs": s["max_abs"],
                "median_channel_abs": s["median_channel_abs"],
                "outlier_ratio": s["outlier_ratio"],
                "worst_token": s["worst_token"],
                "top_channel": s["top_channels"][0],
                "top_magnitude": s["top_magnitudes"][0],
            }))
            for rank, (c, m) in enumerate(
                zip(s["top_channels"], s["top_magnitudes"])
            ):
                channel_rows.append(round_floats({
                    "mask_ratio": key, "layer": layer, "rank": rank,
                    "channel": c, "magnitude": m,
                }))
        print(f"  mask ratio {key}: {len(acts)} layers captured")

    out = pathlib.Path(args.out)
    write_csv(out.with_name(out.name + "_layers.csv"), rows)
    write_csv(out.with_name(out.name + "_channels.csv"), channel_rows)

    report = summarise(per_ratio_top, rows, args)
    print("\n" + report)
    out.with_name(out.name + "_summary.txt").write_text(report, encoding="utf-8")
    out.with_name(out.name + "_top.json").write_text(
        json.dumps(per_ratio_top, indent=2), encoding="utf-8"
    )
    print(f"\n-> {out.name}_layers.csv, {out.name}_channels.csv, "
          f"{out.name}_summary.txt")
    return 0


def summarise(per_ratio_top, rows, args) -> str:
    ratios = list(per_ratio_top)
    layers = sorted(per_ratio_top[ratios[0]])
    lines = ["=" * 68, "ACTIVATION OUTLIERS", "=" * 68]

    # --- how bad, and where -------------------------------------------------
    lines.append("\nOutlier ratio (largest channel / median channel), by depth:")
    lines.append(f"{'layer':>6} " + " ".join(f"{r:>9}" for r in ratios))
    by = {(r["mask_ratio"], r["layer"]): r for r in rows}
    for layer in layers:
        if layer % max(len(layers) // 8, 1) and layer != layers[-1]:
            continue
        cells = " ".join(
            f"{by[(r, layer)]['outlier_ratio']:9.1f}" if (r, layer) in by else " " * 9
            for r in ratios
        )
        lines.append(f"{layer:>6} {cells}")

    # --- same channels across depth? ---------------------------------------
    lines.append("\nAre the outlier channels the same from layer to layer?")
    for r in ratios:
        overlaps = [
            jaccard(per_ratio_top[r][a], per_ratio_top[r][b])
            for a, b in zip(layers, layers[1:])
        ]
        mean = sum(overlaps) / max(len(overlaps), 1)
        lines.append(
            f"  mask ratio {r}: mean overlap between adjacent layers "
            f"{mean:.2f}  ({'stable' if mean > 0.5 else 'shifting'})"
        )

    # --- THE diffusion question --------------------------------------------
    lines.append("\nDo the outlier channels move along the trajectory?")
    lines.append("  (overlap of each layer's top channels vs the all-masked state)")
    base = ratios[0]
    for r in ratios[1:]:
        overlaps = [jaccard(per_ratio_top[base][l], per_ratio_top[r][l])
                    for l in layers]
        mean = sum(overlaps) / max(len(overlaps), 1)
        lines.append(f"  {base} vs {r}: {mean:.2f}")
    lines.append(
        "\n  overlap near 1.0 -> one static rotation fits the whole trajectory;\n"
        "  overlap falling as the sequence decodes -> the outlier set drifts,\n"
        "  and a single rotation is a compromise between the two ends."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
