"""Evaluate a (quantized) DLLM on GSM8K.

Typical A/B, the number the paper's headline claim rests on:

    python scripts/evaluate.py --model GSAI-ML/LLaDA-1.5 --a-bits 16 --w-bits 16
    python scripts/evaluate.py --model GSAI-ML/LLaDA-1.5 --quantize --w-bits 4 --a-bits 4
    python scripts/evaluate.py --model GSAI-ML/LLaDA-1.5 --quantize --w-bits 4 --a-bits 4 --rtn
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from dllmquant.calib.prompts import mix_prompts  # noqa: E402
from dllmquant.config import (  # noqa: E402
    CGQConfig,
    DLLMQuantConfig,
    IAAQConfig,
    QuantConfig,
    RotationConfig,
    TMASConfig,
)
from dllmquant.eval import evaluate_gsm8k  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402
from dllmquant.modules import wrap_linears  # noqa: E402
from dllmquant.pipeline import DLLMQuantPipeline  # noqa: E402
from dllmquant.quantizers import quantize_weight_rtn  # noqa: E402
from dllmquant.report import round_floats, sibling_csv, write_csv  # noqa: E402


def apply_rtn(adapter, cfg: DLLMQuantConfig) -> int:
    """Round-to-nearest baseline: no calibration, no error compensation."""
    n = 0
    for bi, block in enumerate(adapter.blocks):
        layers = wrap_linears(
            block, cfg.weight, cfg.activation, skip=cfg.skip, prefix=f"blocks.{bi}"
        )
        for layer in layers.values():
            layer.set_weight(quantize_weight_rtn(layer.weight.float(), cfg.weight))
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="llada", choices=["llada", "llada2_moe"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--quantize", action="store_true", help="run DLLMQuant first")
    ap.add_argument("--rtn", action="store_true", help="use RTN instead of DLLMQuant")
    ap.add_argument("--w-bits", type=int, default=16)
    ap.add_argument("--a-bits", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=-1)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--nprompts", type=int, default=32)
    ap.add_argument("--no-ia-aq", action="store_true")
    ap.add_argument("--no-cgq-weights", action="store_true",
                    help="ablation: plain GPTQ Hessian, no certainty weighting")
    ap.add_argument("--uniform-calib", action="store_true",
                    help="ablation: uniform sampling instead of TMAS quotas")
    ap.add_argument("--rotate", action="store_true",
                    help="QuaRot rotations (not part of the paper)")
    ap.add_argument("--online-mlp", action="store_true")

    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--gen-length", type=int, default=256)
    ap.add_argument("--eval-steps", type=int, default=256)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cgq = CGQConfig()
    if args.no_cgq_weights:
        # Equal weights for every token == vanilla GPTQ.
        cgq = CGQConfig(unmasked_weight=1.0, masked_base=1.0, beta=0.0)

    cfg = DLLMQuantConfig(
        model_path=args.model,
        model_type=args.model_type,
        dtype=args.dtype,
        device=args.device,
        weight=QuantConfig(
            n_bits=args.w_bits,
            granularity="per_group" if args.group_size > 0 else "per_channel",
            group_size=args.group_size,
            mse_search=True,
        ),
        activation=QuantConfig(n_bits=args.a_bits, granularity="per_token"),
        tmas=TMASConfig(
            n_samples=args.nsamples,
            n_prompts=args.nprompts,
            uniform=args.uniform_calib,
        ),
        cgq=cgq,
        ia_aq=IAAQConfig(enabled=not args.no_ia_aq, n_bits=args.a_bits),
        rotation=RotationConfig(enabled=args.rotate, online_mlp=args.online_mlp),
    )

    adapter = build_adapter(cfg)
    adapter.load()
    print(adapter.describe())

    if args.quantize:
        if args.rtn:
            n = apply_rtn(adapter, cfg)
            print(f"RTN baseline applied to {n} layers")
        else:
            prompts = adapter.encode_prompts(
                mix_prompts([("gsm8k", cfg.tmas.n_prompts)]), max_len=cfg.tmas.seq_len
            )
            report = DLLMQuantPipeline(cfg, adapter).run(prompts)
            print(report.summary())

    gen_cfg = TMASConfig(
        gen_length=args.gen_length, block_length=32, steps=args.eval_steps
    )
    result = evaluate_gsm8k(adapter, n_samples=args.n_eval, gen_cfg=gen_cfg)
    print("\n" + result.summary())

    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "model": args.model, "w_bits": args.w_bits,
                        "a_bits": args.a_bits, "rtn": args.rtn,
                        "ia_aq": not args.no_ia_aq,
                        "certainty_weighting": not args.no_cgq_weights,
                        "rotation": args.rotate,
                    },
                    "accuracy": result.accuracy,
                    "correct": result.correct,
                    "total": result.total,
                    # Every completion, not a sample of them: generation costs
                    # hours and a change to answer extraction must not require
                    # re-running it. scripts/rescore.py re-parses these.
                    "samples": result.samples,
                },
                f,
                indent=2,
            )
        csv_path = write_csv(
            sibling_csv(p),
            [
                round_floats({
                    "question": s["question"].replace("\n", " ")[:200],
                    "predicted": s["pred"],
                    "gold": s["gold"],
                    "correct": s["correct"],
                })
                for s in result.samples
            ],
        )
        print(f"results -> {p}\n         -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
