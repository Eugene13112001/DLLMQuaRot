"""GSM8K on SDAR-8B at one block size, with the committed-block cache rounded on write.

Each block size is its own checkpoint (SDAR trains one per size), so blocks are compared
as trained, not as a knob turned on one model. The sampler is the authors' threshold rule,
greedy (``dllmquant/models/sdar_local.py``); the store rounds the prompt and every finished
block as they are written and never sees the block being decoded
(``dllmquant/committed_cache.py``).

Per question the output keeps: solved or not, answer length, denoising steps, forced
steps, honest bits, and the write check -- the prompt once, then blocks of the block size,
no crops, the store as long as the text. A question that fails the write check is reported;
with ``--strict`` the run stops there, which is how the first run should be made.

    bash scripts/llada2.sh scripts/evaluate_sdar.py --block 32 --bits 16 --n-eval 20 \\
        --strict --out out/sdar_b32_16_n20.json
    bash scripts/llada2.sh scripts/evaluate_sdar.py --block 32 --bits 2 \\
        --key-scale per_channel --out out/sdar_b32_k2pc.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dllmquant.committed_cache import (  # noqa: E402
    KEY_SCALES, RESIDUALS, CommittedCacheConfig, CommittedCacheStats, check_writes,
    make_committed_cache_class,
)
from dllmquant.config import TMASConfig  # noqa: E402
from dllmquant.eval.gsm8k import evaluate_gsm8k  # noqa: E402
from dllmquant.models.sdar_local import (  # noqa: E402
    MASK_ID, SDAR_8B, load_sdar, sdar_generate,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", type=int, required=True, choices=sorted(SDAR_8B))
    ap.add_argument("--model", default="", help="checkpoint (default: SDAR-8B for --block)")
    ap.add_argument("--revision", default=None, help="hub commit to pin; recorded either way")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-eval", type=int, default=1319,
                    help="1319 is all of GSM8K test; 20 and 200 are for checking the run")
    ap.add_argument("--gen-length", type=int, default=2048)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--denoising-steps", type=int, default=0,
                    help="steps per block (0 = block size: at least one token per step)")
    ap.add_argument("--bits", type=int, default=16)
    ap.add_argument("--key-bits", type=int, default=0)
    ap.add_argument("--value-bits", type=int, default=0)
    ap.add_argument("--key-scale", default="per_channel", choices=KEY_SCALES,
                    help="per_channel is KIVI's (one scale per channel per group of tokens); "
                         "per_token is one scale per token per group of channels")
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--residual", default="buffer", choices=RESIDUALS,
                    help="per_channel keys only: keep tokens exact until a group of tokens "
                         "is full (KIVI), or round every block on its own")
    ap.add_argument("--sink-positions", type=int, nargs="*", default=(),
                    help="absolute positions kept exact (SDAR's sink: 2, the third token)")
    ap.add_argument("--strict", action="store_true",
                    help="stop at the first question that fails the write check")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    name = args.model or SDAR_8B[args.block]
    ccfg = CommittedCacheConfig(
        key_bits=args.key_bits or args.bits, value_bits=args.value_bits or args.bits,
        key_scale=args.key_scale, group=args.group, residual=args.residual,
        exact_positions=tuple(args.sink_positions),
    )
    model, tok = load_sdar(name, revision=args.revision, dtype=args.dtype,
                           device=args.device)
    commit = getattr(model.config, "_commit_hash", None)
    print(f"{name} @ {commit}, block {args.block}, threshold {args.threshold}")
    print(f"cache: {ccfg.describe()}")

    from transformers import DynamicCache

    per_q = []

    def generate(prompt, _cfg):
        stats = CommittedCacheStats()
        cache = make_committed_cache_class(DynamicCache, ccfg, stats)()
        out, counts = sdar_generate(
            model, prompt, cache=cache, block_length=args.block,
            gen_length=args.gen_length, denoising_steps=args.denoising_steps or None,
            threshold=args.threshold)
        problems = check_writes(stats, args.block, expected_tokens=counts["stored_tokens"])
        hb = stats.honest_bits(ccfg)
        info = dict(counts, honest_bits_k=hb["k"], honest_bits_v=hb["v"],
                    key_rel_error=stats.key_rel_error(),
                    value_rel_error=stats.value_rel_error(),
                    write_problems=problems)
        per_q.append(info)
        if problems:
            print(f"  write check failed: {problems}")
            if args.strict:
                raise SystemExit(f"--strict: write check failed on question {len(per_q)}")
        return out, info

    adapter = SimpleNamespace(model=model, tokenizer=tok, mask_id=MASK_ID)
    gen_cfg = TMASConfig(gen_length=args.gen_length, block_length=args.block,
                         steps=args.gen_length)
    result = evaluate_gsm8k(adapter, n_samples=args.n_eval, gen_cfg=gen_cfg,
                            generate=generate)
    print("\n" + result.summary())

    n = max(len(per_q), 1)
    stored = sum(q["stored_tokens"] for q in per_q) or 1
    summary = {
        "steps_mean": sum(q["steps"] for q in per_q) / n,
        "forced_steps_share": sum(q["forced_steps"] for q in per_q)
                              / max(sum(q["steps"] for q in per_q), 1),
        "tokens_per_step": sum(q["committed"] for q in per_q)
                           / max(sum(q["steps"] for q in per_q), 1),
        "completion_tokens_mean": sum(s["completion_tokens"] for s in result.samples) / n,
        # weighted by stored tokens: what the store costs over the whole run
        "honest_bits_k": sum(q["honest_bits_k"] * q["stored_tokens"] for q in per_q) / stored,
        "honest_bits_v": sum(q["honest_bits_v"] * q["stored_tokens"] for q in per_q) / stored,
        "key_rel_error_mean": sum(q["key_rel_error"] for q in per_q) / n,
        "write_check_failures": sum(1 for q in per_q if q["write_problems"]),
    }
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "model": name, "revision": args.revision, "commit": commit,
                    "block": args.block, "gen_length": args.gen_length,
                    "threshold": args.threshold,
                    "denoising_steps": args.denoising_steps or args.block,
                    "greedy": True, "n_eval": args.n_eval, "dtype": args.dtype,
                    "key_bits": ccfg.key_bits, "value_bits": ccfg.value_bits,
                    "key_scale": ccfg.key_scale, "value_scale": "per_token",
                    "group": ccfg.group, "residual": ccfg.residual,
                    "sink_positions": list(ccfg.exact_positions),
                    "torch": torch.__version__,
                },
                "accuracy": result.accuracy, "correct": result.correct,
                "total": result.total, "cut_off": result.cut_off,
                "summary": summary, "samples": result.samples,
            }, f, indent=2)
        print(f"results -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
