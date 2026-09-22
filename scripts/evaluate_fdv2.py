"""GSM8K on Fast-dLLM-v2 with a quantized prefix cache, in the checkpoint's own sampler.

The third block dLLM of this project, and the first one evaluated in the harness the
BitSieve tables use: blocks of 32, sub-blocks of 8, commit every masked position at or
above the confidence threshold. Nothing about decoding is reimplemented here -- the model's
own ``generate`` runs, and the only thing replaced is the class its cache is built from
(``dllmquant/models/fast_dllm_v2_local.py``).

Why this model. It has no QK-Norm and a bias on ``k_proj``, so if the key axis still decides
the cache at two bits, the fixed-channel outliers come from the bias, not from a norm gain;
and if it does not, the claim from LLaDA2.0-mini is bounded rather than general. Either way
the first cell to run is the lossless one: it has to reproduce the published dense number
(85.0 on GSM8K, n = 200) before any quantized cell means anything.

    bash scripts/llada2.sh scripts/evaluate_fdv2.py --model Efficient-Large-Model/Fast_dLLM_v2_7B \\
        --kv-bits 16 --out out/fd_16.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dllmquant.config import DLLMQuantConfig, QuantConfig, TMASConfig  # noqa: E402
from dllmquant.eval.gsm8k import evaluate_gsm8k  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402
from dllmquant.models.fast_dllm_v2_local import (  # noqa: E402
    fdv2_generate, install_quantized_cache,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--gen-length", type=int, default=2048,
                    help="the published setting for the long-generation regime")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--small-block-size", type=int, default=8)
    ap.add_argument("--kv-bits", type=int, default=16)
    ap.add_argument("--kv-key-bits", type=int, default=0)
    ap.add_argument("--kv-value-bits", type=int, default=0)
    ap.add_argument("--kv-group-size", type=int, default=128)
    ap.add_argument("--kv-key-axis", default="token", choices=["token", "channel"],
                    help="grouping direction: 'token' is one scale per channel (KIVI's "
                         "per-channel, the axis this family's caches use), 'channel' is "
                         "one scale per token")
    ap.add_argument("--kv-value-axis", default="channel", choices=["token", "channel"])
    ap.add_argument("--kv-value-group-size", type=int, default=0,
                    help="channels per V scale (0 = --kv-group-size). BitSieve uses 32: one "
                         "scale per token per 32 channels, where this project's default is "
                         "the whole 128-channel head")
    ap.add_argument("--kv-key-noise", type=float, default=0.0,
                    help="structureless control for thesis 4: Gaussian noise on the stored keys, in units of the RMS of what is written, per head. At --kv-bits 16 it is the only error the cache carries, so a dose matched on the centered logit error asks whether a model dies of the size of the error or of its shape")
    ap.add_argument("--kv-clip", type=float, default=0.95,
                    help="fraction of the min-max range kept (1.0 = plain min-max, as in "
                         "BitSieve; 0.95 is this project's default)")
    ap.add_argument("--pre-bias", action="store_true",
                    help="store K with the rotated k_proj bias removed and add it back on "
                         "read. Exact at any width, and it takes the parameter-induced part "
                         "of the key outlier out of the quantizer's range -- the counterpart "
                         "of the QK-Norm gain migration on a model with no norm to migrate")
    ap.add_argument("--key-mean", action="store_true",
                    help="SageAttention's smooth K on the store: subtract the per-channel mean "
                         "of the keys each write stores, quantize, add it back. The data-driven "
                         "baseline for --pre-bias")
    ap.add_argument("--rotate-qk", action="store_true",
                    help="R4: rotate Q and K head-wise after RoPE, so the store holds "
                         "rotated keys -- QuaRot's arrangement, to be read together with "
                         "--kv-key-axis channel")
    ap.add_argument("--use-block-cache", action="store_true",
                    help="the checkpoint's own reuse of the current block: computed in full at "
                         "each sub-block start, then only the sub-block recomputed while the "
                         "rest of the block is read back stale. Off is what the published "
                         "numbers use. With a width below 16 the stale entries are rounded "
                         "too -- the staleness x width pair of LLaDA2.0-mini's policy cells")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = DLLMQuantConfig(
        model_path=args.model, model_type="fast_dllm_v2",
        dtype=args.dtype, device=args.device,
        weight=QuantConfig(n_bits=16), activation=QuantConfig(n_bits=16),
        tmas=TMASConfig(gen_length=args.gen_length, block_length=args.block_size,
                        steps=args.gen_length),
    )
    adapter = build_adapter(cfg)
    adapter.load()
    print(adapter.describe())

    if args.rotate_qk:
        from dllmquant.algos.quarot import install_qk_rotation
        install_qk_rotation(adapter)
        print("R4 installed: Q and K rotated head-wise after RoPE")

    remove, stats = install_quantized_cache(
        adapter, bits=args.kv_bits, key_bits=args.kv_key_bits,
        value_bits=args.kv_value_bits, group_size=args.kv_group_size,
        key_axis=args.kv_key_axis, value_axis=args.kv_value_axis,
        pre_bias=args.pre_bias, clip_ratio=args.kv_clip, key_noise=args.kv_key_noise,
        value_group_size=args.kv_value_group_size, key_mean=args.key_mean,
    )
    print(f"prefix cache at {args.kv_bits} bits, group {args.kv_group_size}, "
          f"K along {args.kv_key_axis}, V along {args.kv_value_axis}")

    gen_cfg = TMASConfig(gen_length=args.gen_length, block_length=args.block_size,
                         steps=args.gen_length)

    def generate(prompt, cfg_):
        return fdv2_generate(adapter, prompt, cfg_, threshold=args.threshold,
                             block_size=args.block_size,
                             small_block_size=args.small_block_size,
                             use_block_cache=args.use_block_cache)

    try:
        result = evaluate_gsm8k(adapter, n_samples=args.n_eval, gen_cfg=gen_cfg,
                                generate=generate)
    finally:
        remove()
    print("\n" + result.summary())
    print("  " + stats.describe())

    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "model": args.model, "model_type": "fast_dllm_v2",
                    "code_revision": getattr(adapter, "code_revision", None),
                    "gen_length": args.gen_length, "n_eval": args.n_eval,
                    "threshold": args.threshold, "block_size": args.block_size,
                    "small_block_size": args.small_block_size,
                    "kv_cache": True, "kv_bits": args.kv_bits,
                    "kv_key_bits": args.kv_key_bits or None,
                    "kv_value_bits": args.kv_value_bits or None,
                    "kv_group_size": args.kv_group_size,
                    "kv_key_axis": args.kv_key_axis,
                    "kv_value_axis": args.kv_value_axis,
                    "rotate_qk": args.rotate_qk, "pre_bias": args.pre_bias,
                    "key_mean": args.key_mean,
                    "kv_clip": args.kv_clip, "use_block_cache": args.use_block_cache,
                    "kv_key_noise": args.kv_key_noise,
                    "kv_value_group_size": args.kv_value_group_size or args.kv_group_size,
                    "prefix_writes": stats.writes, "prefix_entries": stats.entries,
                },
                "accuracy": result.accuracy, "correct": result.correct,
                "total": result.total, "cut_off": result.cut_off,
                "cut_off_wrong": result.cut_off_wrong,
                "samples": result.samples,
            }, f, indent=2)
        print(f"results -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
