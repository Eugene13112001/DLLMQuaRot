"""Quantize a diffusion LLM with DLLMQuant.

    python scripts/quantize.py \
        --model GSAI-ML/LLaDA-1.5 --model-type llada \
        --w-bits 4 --a-bits 4 --nsamples 128 \
        --save out/llada15-w4a4

Multi-GPU: the model is loaded with device_map="auto", and blocks are solved
one at a time, so peak extra memory is one block's Hessians -- not the model.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dllmquant.calib.prompts import load_prompts, mix_prompts  # noqa: E402
from dllmquant.config import (  # noqa: E402
    CGQConfig,
    DLLMQuantConfig,
    IAAQConfig,
    QuantConfig,
    RotationConfig,
    TMASConfig,
    quarot_baseline,
    quarot_diffusion,
    quarot_w4a4,
)
from dllmquant.models import build_adapter  # noqa: E402
from dllmquant.models.base import estimate_required_gb, preflight_memory  # noqa: E402
from dllmquant.pipeline import DLLMQuantPipeline  # noqa: E402
from dllmquant.report import round_floats, sibling_csv, write_csv  # noqa: E402


def activation_config(args) -> QuantConfig:
    """Per-token by default; per-group when asked.

    Per-token gives every token its own scale but shares it across all 4096
    channels, so one outlier channel sets the step size for the whole token.
    Grouping splits that: one scale per (token, group of channels). It is the
    fallback when rotation alone does not flatten the channel axis enough --
    and unlike rotation it costs real inference work, since the scales are
    recomputed per forward.
    """
    if args.a_group_size > 0:
        return QuantConfig(
            n_bits=args.a_bits,
            granularity="per_group",
            group_size=args.a_group_size,
            dynamic=True,
        )
    return QuantConfig(n_bits=args.a_bits, granularity="per_token")


def build_config(args) -> DLLMQuantConfig:
    if args.recipe == "quarot-full":
        # Everything: rotations first, then all three of the paper's
        # components calibrated in the rotated basis.
        cfg = quarot_w4a4()
        cfg.model_path = args.model
        cfg.model_type = args.model_type
        cfg.dtype, cfg.device, cfg.seed = args.dtype, args.device, args.seed
        cfg.device_map = args.device_map or None
        cfg.weight.n_bits = args.w_bits
        cfg.activation.n_bits = args.a_bits
        cfg.ia_aq.n_bits = args.a_bits
        cfg.tmas.n_samples = args.nsamples
        cfg.tmas.n_prompts = args.nprompts
        cfg.tmas.seq_len = args.seq_len
        cfg.tmas.proportions = tuple(args.proportions)
        cfg.rotation.online_mlp = not args.no_online_mlp
        cfg.checkpoint_dir = args.checkpoint_dir
        cfg.max_group_layers = args.max_group_layers
        cfg.max_blocks = args.max_blocks
        cfg.rotate_only = args.rotate_only
        return cfg

    if args.recipe in ("quarot-baseline", "quarot-diffusion"):
        # QuaRot without any DLLMQuant component. The two differ only in where
        # GPTQ's calibration data comes from: plain text (the paper's baseline,
        # i.e. QuaRot ported unchanged) or real denoising states (adapted).
        cfg = (
            quarot_baseline() if args.recipe == "quarot-baseline"
            else quarot_diffusion()
        )
        cfg.model_path = args.model
        cfg.model_type = args.model_type
        cfg.dtype, cfg.device, cfg.seed = args.dtype, args.device, args.seed
        cfg.weight.n_bits = args.w_bits
        cfg.activation.n_bits = args.a_bits
        cfg.ia_aq.n_bits = args.a_bits
        cfg.tmas.n_samples = args.nsamples
        cfg.tmas.n_prompts = args.nprompts
        cfg.tmas.seq_len = args.seq_len
        cfg.rotation.online_mlp = not args.no_online_mlp
        cfg.device_map = args.device_map or None
        cfg.checkpoint_dir = args.checkpoint_dir
        cfg.max_group_layers = args.max_group_layers
        cfg.max_blocks = args.max_blocks
        cfg.rotate_only = args.rotate_only
        return cfg

    return DLLMQuantConfig(
        model_path=args.model,
        model_type=args.model_type,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map or None,
        checkpoint_dir=args.checkpoint_dir,
        max_group_layers=args.max_group_layers,
        max_blocks=args.max_blocks,
        rotate_only=args.rotate_only,
        seed=args.seed,
        weight=QuantConfig(
            n_bits=args.w_bits,
            symmetric=args.w_symmetric,
            granularity="per_group" if args.group_size > 0 else "per_channel",
            group_size=args.group_size,
            mse_search=not args.no_weight_mse,
        ),
        activation=activation_config(args),
        tmas=TMASConfig(
            n_samples=args.nsamples,
            n_prompts=args.nprompts,
            seq_len=args.seq_len,
            gen_length=args.gen_length,
            block_length=args.block_length,
            steps=args.steps,
            proportions=tuple(args.proportions),
            uniform=args.uniform_calib,
            mode=args.calib,
            seed=args.seed,
        ),
        cgq=CGQConfig(
            unmasked_weight=args.unmasked_weight,
            masked_base=args.masked_base,
            beta=args.beta,
            percdamp=args.percdamp,
            act_order=not args.no_act_order,
            static_groups=args.static_groups,
        ),
        ia_aq=IAAQConfig(
            enabled=not args.no_ia_aq,
            n_bits=args.a_bits,
            weight_power=args.weight_power,
        ),
        rotation=RotationConfig(
            # --rotate-only implies it. The two used to be independent, so
            # `--rotate-only` on its own applied nothing, returned in 0.0s and
            # saved an *unrotated* copy under whatever name was asked for --
            # and the run that then measured it compared a model against
            # itself. A flag whose name says "rotate" must rotate.
            enabled=args.rotate or args.rotate_only,
            value_heads=not args.no_value_rotation,
            online_mlp=args.online_mlp,
            seed=args.seed,
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="llada", choices=["llada", "llada2_moe"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default="",
                    help="only when the model does not fit on one GPU "
                         "(e.g. 'auto' for LLaDA2.0-flash)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rotate-only", action="store_true",
                    help="apply QuaRot and stop, weights left unquantized. "
                         "Produces the rotated fp16 model the cache sweeps "
                         "need: rotation reads no data, so this is minutes, "
                         "where --rotate --w-bits 16 would still run the "
                         "solver over every layer for weights it discards.")
    ap.add_argument("--max-blocks", type=int, default=0,
                    help="stop after N blocks. Debugging only -- the saved "
                         "model would be half quantized and must not be "
                         "evaluated. Use it to ask a question about the run's "
                         "bookkeeping in minutes instead of the 20 hours the "
                         "full solve costs; one block exercises every path.")
    ap.add_argument("--max-group-layers", type=int, default=64,
                    help="layers of one group calibrated at a time; each holds "
                         "an in_features^2 Hessian, so a 512-expert group asks "
                         "for 8.6 GB at once. Lower it on a shared card; the "
                         "result does not depend on it")

    ap.add_argument(
        "--recipe",
        default="dllmquant",
        choices=["dllmquant", "quarot-full", "quarot-baseline", "quarot-diffusion"],
        help="'dllmquant' = the paper's three components. 'quarot-full' = those "
             "plus rotations, in QuaRot's numeric settings. 'quarot-baseline' = "
             "QuaRot alone with text calibration (the paper's baseline row); "
             "'quarot-diffusion' = QuaRot alone calibrated on denoising states. "
             "The three quarot-* recipes ignore the component flags below.",
    )
    ap.add_argument("--calib", default="trajectory",
                    choices=["trajectory", "text"],
                    help="'text' calibrates on plain text, with no masks at all")
    ap.add_argument("--no-online-mlp", action="store_true")

    g = ap.add_argument_group("format")
    g.add_argument("--w-bits", type=int, default=4)
    g.add_argument("--a-bits", type=int, default=4)
    g.add_argument("--group-size", type=int, default=-1,
                   help="weight group size; -1 = per output channel")
    g.add_argument("--a-group-size", type=int, default=-1,
                   help="activation group size: one scale per N channels "
                        "instead of one per token. The second lever against "
                        "channel outliers after rotation; -1 = per token")
    g.add_argument("--w-symmetric", action="store_true")
    g.add_argument("--no-weight-mse", action="store_true")

    g = ap.add_argument_group("TMAS")
    g.add_argument("--nsamples", type=int, default=128)
    g.add_argument("--nprompts", type=int, default=32)
    g.add_argument("--seq-len", type=int, default=256)
    g.add_argument("--gen-length", type=int, default=128)
    g.add_argument("--block-length", type=int, default=32)
    g.add_argument("--steps", type=int, default=64)
    g.add_argument("--proportions", type=float, nargs=4,
                   default=[0.3, 0.2, 0.2, 0.3])
    g.add_argument("--uniform-calib", action="store_true",
                   help="ablation: sample the trajectory uniformly, no quotas")
    g.add_argument("--calib-dataset", default="gsm8k")

    g = ap.add_argument_group("CGQ")
    g.add_argument("--unmasked-weight", type=float, default=0.5)
    g.add_argument("--masked-base", type=float, default=0.5)
    g.add_argument("--beta", type=float, default=1.0)
    g.add_argument("--percdamp", type=float, default=0.01)
    g.add_argument("--no-act-order", action="store_true")
    g.add_argument("--static-groups", action="store_true")

    g = ap.add_argument_group("IA-AQ")
    g.add_argument("--no-ia-aq", action="store_true")
    g.add_argument("--weight-power", type=float, default=1.0)

    g = ap.add_argument_group("QuaRot (not part of the paper; off by default)")
    g.add_argument("--rotate", action="store_true",
                   help="orthogonal rotations against channel-wise outliers")
    g.add_argument("--no-value-rotation", action="store_true")
    g.add_argument("--online-mlp", action="store_true",
                   help="online Hadamard before the MLP down-projection")

    ap.add_argument("--save", default="")
    ap.add_argument("--report", default="")
    ap.add_argument("--force", action="store_true",
                    help="start even if the GPU looks too full")
    ap.add_argument("--checkpoint-dir", default="",
                    help="save each finished block here; a restart with the "
                         "same directory resumes instead of starting over "
                         "(costs ~14 GB of disk for LLaDA-8B)")
    args = ap.parse_args()

    cfg = build_config(args)
    print(json.dumps({
        "recipe": args.recipe,
        "model": cfg.model_path, "w_bits": cfg.weight.n_bits,
        "a_bits": cfg.activation.n_bits, "nsamples": cfg.tmas.n_samples,
        "calibration": cfg.tmas.mode,
        "proportions": list(cfg.tmas.proportions) if cfg.tmas.mode == "trajectory"
                       else None,
        "ia_aq": cfg.ia_aq.enabled,
        "certainty_weighting": cfg.cgq.beta > 0,
        "rotation": cfg.rotation.enabled,
        "act_clip_ratio": cfg.activation.clip_ratio,
    }, indent=2))

    # Fail in two seconds on a full GPU, not twenty minutes in -- but only when
    # a GPU is what the job will use. A CPU run needs no VRAM at all, and this
    # check was refusing to start one on a node whose cards were busy, while
    # 1.8 TB of host RAM sat idle beside them.
    if cfg.device_map or (cfg.device and cfg.device != "cpu"):
        preflight_memory(
            estimate_required_gb(cfg), strict=not args.force,
            device_map=cfg.device_map,
        )

    adapter = build_adapter(cfg)
    adapter.load()

    prompts_text = mix_prompts(
        [(args.calib_dataset, cfg.tmas.n_prompts)], seed=cfg.seed
    ) if args.calib_dataset else load_prompts(cfg.tmas.n_prompts, dataset=None)
    prompts = adapter.encode_prompts(prompts_text, max_len=cfg.tmas.seq_len)

    t0 = time.time()
    pipeline = DLLMQuantPipeline(cfg, adapter)
    report = pipeline.run(prompts)
    print("\n" + report.summary())
    print(f"total {time.time() - t0:.0f}s")

    if args.report:
        pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "layers": [vars(l) for l in report.layers],
                    "ia_aq": report.ia_aq,
                    "n_calibration": report.n_calibration,
                    "unweighted_layers": report.unweighted_layers,
                    "seconds": report.seconds,
                },
                f,
                indent=2,
            )
        # Per-layer CSV: this is the table you actually sort when hunting for
        # the layer that took the most damage.
        csv_path = write_csv(
            sibling_csv(args.report),
            [
                round_floats({
                    "layer": l.name,
                    "block": l.block,
                    "proxy_loss": l.proxy_loss,
                    "calib_tokens": l.tokens,
                    "certainty_weighted": l.weighted,
                    "seconds": l.seconds,
                })
                for l in report.layers
            ],
        )
        print(f"report -> {args.report}\n       -> {csv_path}")

    if args.save:
        out = pathlib.Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        # Fake-quant weights are stored in the compute dtype: the values sit on
        # the 4-bit grid but each still occupies 2 bytes.  This checkpoint
        # measures accuracy; it does not shrink on disk.  See README.
        adapter.model.save_pretrained(out, safe_serialization=True)
        adapter.tokenizer.save_pretrained(out)
        print(f"model -> {out}  (simulated quantization, {cfg.dtype} storage)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
