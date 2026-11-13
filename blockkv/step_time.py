"""Time of one denoising step of SDAR (uncompressed, real milliseconds) against cache
length, block and batch, and what a real 2-bit cache could save.

    python -m blockkv.step_time --block 32 --lengths 1024 4096 8192 16384 --batch 1 4 \
        --out out/timing/sdar8b_b32.jsonl

Per (length, batch): the cache is filled with `length` random tokens (contents do not
matter for time), then the step forward (current block against the cache, nothing stored)
is timed `--reps` times after `--warmup` and averaged.

The cost model, per batch: step time = fixed part + slope * cache bytes, a least-squares
line over the lengths. The slope is the price of reading the cache. A real cache stored
at `--bits` honest bits per number reads bits/16 of the bytes, so
    t_2bit = t_16 - slope * cache_bytes * (1 - bits / 16)
which assumes free dequantization: an upper bound on the saving. `break_even` is the
step tax at which the saving is gone: t_16 / t_2bit - 1. A step tax larger than that
makes the compressed model slower per answer, smaller makes it faster.
"""
import argparse
import json
import time
from pathlib import Path

import torch


def cache_bytes(cache):
    from blockkv.kvquant import cache_layers
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in cache_layers(cache))


def fit_line(xs, ys):
    """Least-squares y = a + b x."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return my - b * mx, b


def cost_model(rows, bits=3.0):
    """Add per row: ms_fixed, ms_per_gb (from a line over the lengths of the same batch and
    block), est_ms_2bit and break_even. Rows: dicts with block, batch, cache_bytes, ms_step."""
    out = []
    groups = {}
    for r in rows:
        groups.setdefault((r["block"], r["batch"]), []).append(r)
    for (_, _), rs in sorted(groups.items()):
        a, b = fit_line([r["cache_bytes"] / 2 ** 30 for r in rs], [r["ms_step"] for r in rs])
        for r in rs:
            gb = r["cache_bytes"] / 2 ** 30
            saved = max(b, 0.0) * gb * (1 - bits / 16)
            t2 = max(r["ms_step"] - saved, 1e-9)
            out.append(dict(r, ms_fixed=round(a, 3), ms_per_gb=round(b, 3), cache_gb=round(gb, 3),
                            est_ms_2bit=round(t2, 3), est_speedup=round(r["ms_step"] / t2, 3),
                            break_even=round(r["ms_step"] / t2 - 1, 4)))
    return out


@torch.no_grad()
def time_step(model, length, block, batch, reps, warmup, chunk=1024):
    from transformers.cache_utils import DynamicCache
    dev = next(model.parameters()).device
    cache = DynamicCache()
    gen = torch.Generator(device="cpu").manual_seed(0)
    ids = torch.randint(1000, 30000, (batch, length), generator=gen).to(dev)
    step = max(block, chunk // block * block)
    for lo in range(0, length, step):
        hi = min(lo + step, length)
        model(ids[:, lo:hi], attention_mask=torch.ones(1, hi - lo, hi, device=dev),
              position_ids=torch.arange(lo, hi, device=dev).unsqueeze(0).expand(batch, -1),
              past_key_values=cache, use_cache=True, store_kv=True)
    cur = torch.randint(1000, 30000, (batch, block), generator=gen).to(dev)
    mask = torch.ones(1, block, length + block, device=dev)
    pos = torch.arange(length, length + block, device=dev).unsqueeze(0).expand(batch, -1)

    def fwd():
        model(cur, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True, store_kv=False)

    for _ in range(warmup):
        fwd()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fwd()
    torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t0) / reps
    return ms, cache_bytes(cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sdar8b", choices=["sdar8b", "sdar30b"])
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--bits", type=float, default=3.0, help="honest bits per number of the real compressed cache")
    ap.add_argument("--model-path")
    ap.add_argument("--torch-rmsnorm", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    from blockkv.run_gsm8k import device_info, patch_rmsnorm, repo, resolve_path

    _, repo_id = repo(args.model, args.block)
    model = AutoModelForCausalLM.from_pretrained(
        resolve_path(repo_id, args.model_path), torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda", low_cpu_mem_usage=True).eval()
    if args.torch_rmsnorm and not patch_rmsnorm(model):
        raise SystemExit("--torch-rmsnorm: no flash_rms_norm in this model's code")

    rows = []
    for batch in args.batch:
        for length in args.lengths:
            try:
                ms, nbytes = time_step(model, length, args.block, batch, args.reps, args.warmup)
            except torch.OutOfMemoryError:
                print(f"length {length} batch {batch}: out of memory, skipped", flush=True)
                torch.cuda.empty_cache()
                continue
            rows.append(dict(model=args.model, block=args.block, length=length, batch=batch,
                             ms_step=round(ms, 3), cache_bytes=nbytes))
            print(f"block {args.block} length {length} batch {batch}: {ms:.2f} ms/step, cache {nbytes / 2**30:.2f} GiB", flush=True)
            torch.cuda.empty_cache()

    rows = cost_model(rows, args.bits)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(vars(args), repo=repo_id, **device_info())
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{'len':>6s} {'batch':>5s} {'ms/step':>8s} {'cacheGiB':>8s} {'ms/GiB':>7s} {'2bit ms':>8s} {'speedup':>7s} {'break-even tax':>14s}")
    for r in rows:
        print(f"{r['length']:6d} {r['batch']:5d} {r['ms_step']:8.2f} {r['cache_gb']:8.2f} {r['ms_per_gb']:7.2f} "
              f"{r['est_ms_2bit']:8.2f} {r['est_speedup']:7.3f} {100 * r['break_even']:13.1f}%")


if __name__ == "__main__":
    main()
