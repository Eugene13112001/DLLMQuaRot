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
    ap.add_argument("--group-size", type=int, default=-1,
                    help="weight group size; -1 = per output channel")
    ap.add_argument("--a-group-size", type=int, default=-1,
                    help="activation group size: one scale per N channels "
                         "instead of one per token; -1 = per token")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--max-group-layers", type=int, default=64,
                    help="layers of one group calibrated at a time; each holds "
                         "an in_features^2 Hessian, so a 512-expert group asks "
                         "for 8.6 GB at once. Lower it on a shared card; the "
                         "result does not depend on it")

    ap.add_argument("--nprompts", type=int, default=32)
    ap.add_argument("--no-ia-aq", action="store_true")
    ap.add_argument("--no-cgq-weights", action="store_true",
                    help="ablation: plain GPTQ Hessian, no certainty weighting")
    ap.add_argument("--uniform-calib", action="store_true",
                    help="ablation: uniform sampling instead of TMAS quotas")
    ap.add_argument("--rotate", action="store_true",
                    help="QuaRot rotations (not part of the paper)")
    ap.add_argument("--online-mlp", action="store_true")

    g = ap.add_argument_group("KV cache (LLaDA2.0 only; block diffusion makes "
                              "the prefix exactly reusable)")
    g.add_argument("--kv-cache", action="store_true",
                   help="read the prefix from a quantized cache instead of "
                        "recomputing it at every denoising step")
    g.add_argument("--kv-bits", type=int, default=4)
    g.add_argument("--kv-key-axis", default="token",
                   choices=["channel", "token"],
                   help="direction a K group runs in. Measured to matter more "
                        "than any other knob on this cache: along channels "
                        "four bits cost twelve points of committed decisions, "
                        "along tokens nothing at all, because K's outliers sit "
                        "in fixed channels and a group must not straddle them. "
                        "Defaults to the better one -- the analytic sweeps "
                        "default to 'channel' only so their earlier numbers "
                        "stay reproducible, and there is no such history here.")
    g.add_argument("--kv-value-axis", default="channel",
                   choices=["channel", "token"],
                   help="the same for V, where it was measured to make almost "
                        "no difference")
    g.add_argument("--kv-group-size", type=int, default=128,
                   help="channels sharing one scale along head_dim; 128 is "
                        "the whole head on LLaDA2.0-mini")
    g.add_argument("--kv-key-bits", type=int, default=0,
                   help="override the width for K only (0 = same as --kv-bits)")
    g.add_argument("--kv-value-bits", type=int, default=0,
                   help="override the width for V only (0 = same as --kv-bits)")
    g.add_argument("--kv-masked-bits", type=int, default=0,
                   help="width for positions still carrying the mask token "
                        "(0 = same as --kv-bits). Their K/V is overwritten as "
                        "soon as the token is committed, so precision spent "
                        "there has a short shelf life -- which is a claim this "
                        "flag exists to test rather than assume")

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
        activation=(
            QuantConfig(n_bits=args.a_bits, granularity="per_group",
                        group_size=args.a_group_size, dynamic=True)
            if args.a_group_size > 0
            else QuantConfig(n_bits=args.a_bits, granularity="per_token")
        ),
        tmas=TMASConfig(
            n_samples=args.nsamples,
            n_prompts=args.nprompts,
            uniform=args.uniform_calib,
        ),
        cgq=cgq,
        ia_aq=IAAQConfig(enabled=not args.no_ia_aq, n_bits=args.a_bits),
        rotation=RotationConfig(enabled=args.rotate, online_mlp=args.online_mlp),
        max_group_layers=args.max_group_layers,
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

    generate = None
    if args.kv_cache:
        if args.model_type != "llada2_moe":
            raise SystemExit(
                "--kv-cache needs block-causal attention. LLaDA-1.5 attends "
                "over the whole sequence in both directions, so no position's "
                "K/V is ever final and there is nothing to reuse."
            )
        from dllmquant.cache import BlockKVCache, KVCacheConfig
        from dllmquant.models.llada2_local import cached_generate

        kv_cfg = KVCacheConfig(
            enabled=True,
            decoded_bits=args.kv_bits,
            masked_bits=args.kv_masked_bits or args.kv_bits,
            key_bits=args.kv_key_bits or None,
            value_bits=args.kv_value_bits or None,
            group_size=args.kv_group_size,
            key_axis=args.kv_key_axis,
            value_axis=args.kv_value_axis,
        )
        n_layers = len(adapter.blocks)
        print(f"\nKV cache on: {args.kv_bits} bits, group {args.kv_group_size}"
              + (f", K at {args.kv_key_bits}" if args.kv_key_bits else "")
              + (f", V at {args.kv_value_bits}" if args.kv_value_bits else ""))

        def generate(prompt, cfg_):
            # A cache per question: entries are about this sequence and
            # carrying them across would be measuring a different thing.
            return cached_generate(adapter, prompt, cfg_,
                                   BlockKVCache(kv_cfg, n_layers))

    result = evaluate_gsm8k(
        adapter, n_samples=args.n_eval, gen_cfg=gen_cfg, generate=generate
    )
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
                        "a_group_size": args.a_group_size,
                        # Weights had no entry here while activations did, so a
                        # finished run could not be asked what granularity its
                        # weights used -- the one question that decides whether
                        # two results belong on the same axis.
                        "group_size": args.group_size,
                        # The generation budget belongs in the record: two runs
                        # at different gen_length are not comparable, and a
                        # file that does not say which one it was cannot be
                        # placed next to another one later.
                        "gen_length": args.gen_length,
                        "eval_steps": args.eval_steps,
                        "n_eval": args.n_eval,
                        "kv_cache": args.kv_cache,
                        "kv_bits": args.kv_bits if args.kv_cache else None,
                        "kv_group_size": args.kv_group_size if args.kv_cache else None,
                        "kv_key_axis": args.kv_key_axis if args.kv_cache else None,
                        "kv_value_axis": args.kv_value_axis if args.kv_cache else None,
                        "kv_key_bits": args.kv_key_bits or None,
                        "kv_value_bits": args.kv_value_bits or None,
                        "kv_masked_bits": args.kv_masked_bits or None,
                    },
                    "accuracy": result.accuracy,
                    "correct": result.correct,
                    "total": result.total,
                    "cut_off": result.cut_off,
                    "cut_off_wrong": result.cut_off_wrong,
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
