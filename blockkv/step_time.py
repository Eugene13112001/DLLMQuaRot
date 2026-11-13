"""Time of one denoising step of SDAR (uncompressed, real milliseconds) against cache
length, block and batch, and what a real 2-bit cache could save.

    python -m blockkv.step_time --block 32 --lengths 1024 4096 16384 --batch 1 4 16 \
        --rounds 5 --out out/timing/sdar8b_b32.jsonl

Per (length, batch): the cache is filled with `length` random tokens (contents do not
matter for time), then the step forward (current block against the cache, nothing
stored) is timed `--reps` times after `--warmup`, and so is the commit forward (the
block written into the cache, once per block). Points are measured in `--rounds`
interleaved rounds and the minimum is kept: neighbours on a shared GPU can only slow a
step down, so the minimum is closest to an idle GPU; the spread between rounds is kept
too.

Two estimates for a real cache at `--bits` honest bits per number:
  measured cost model, per (block, batch): step time = fixed + slope * cache bytes, a
    least-squares line over the lengths; the slope is the price of reading the cache,
    and a compressed cache reads bits/16 of the bytes:
        t_2bit = t_16 - slope * cache_bytes * (1 - bits / 16)
    (free dequantization: an upper bound on the saving of this implementation);
  memory-bound ideal: a step reads the weights and the cache once at `--bandwidth`:
        t = (weight_bytes + cache_bytes) / bandwidth, and with bits/16 of the cache bytes.
`break_even` is the step tax at which the saving is gone: t_16 / t_2bit - 1. A larger
step tax makes the compressed model slower per answer, a smaller one faster.
"""
import argparse
import json
import subprocess
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


def cost_model(rows, bits=3.0, weight_bytes=None, bandwidth_tbs=None):
    """Add per row: ms_fixed, ms_per_gb (a line over the lengths of the same block and
    batch), est_ms_2bit, est_speedup, break_even; and, given weight bytes and bandwidth,
    the memory-bound ideal (ideal_ms_16, ideal_ms_2bit, ideal_speedup, ideal_break_even).
    Rows: dicts with block, batch, cache_bytes, ms_step."""
    out = []
    groups = {}
    for r in rows:
        groups.setdefault((r["block"], r["batch"]), []).append(r)
    for _, rs in sorted(groups.items()):
        a, b = fit_line([r["cache_bytes"] / 2 ** 30 for r in rs], [r["ms_step"] for r in rs])
        for r in rs:
            gb = r["cache_bytes"] / 2 ** 30
            saved = max(b, 0.0) * gb * (1 - bits / 16)
            t2 = max(r["ms_step"] - saved, 1e-9)
            row = dict(r, ms_fixed=round(a, 3), ms_per_gb=round(b, 3), cache_gb=round(gb, 3),
                       est_ms_2bit=round(t2, 3), est_speedup=round(r["ms_step"] / t2, 3),
                       break_even=round(r["ms_step"] / t2 - 1, 4))
            if weight_bytes and bandwidth_tbs:
                bw = bandwidth_tbs * 1e12 / 1e3            # bytes per ms
                i16 = (weight_bytes + r["cache_bytes"]) / bw
                i2 = (weight_bytes + r["cache_bytes"] * bits / 16) / bw
                row.update(ideal_ms_16=round(i16, 3), ideal_ms_2bit=round(i2, 3),
                           ideal_speedup=round(i16 / i2, 3), ideal_break_even=round(i16 / i2 - 1, 4))
            out.append(row)
    return out


def gpu_util():
    """Utilization (%) of the visible GPU from nvidia-smi, matched by UUID; None if unknown."""
    try:
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        res = subprocess.run(["nvidia-smi", "--query-gpu=uuid,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        for line in res.splitlines():
            u, util = [s.strip() for s in line.split(",")]
            if u.replace("GPU-", "").startswith(uuid[:8]):
                return int(util)
    except Exception:
        return None
    return None


@torch.no_grad()
def time_point(model, length, block, batch, reps, warmup, chunk=1024):
    """(ms per step, ms per commit, cache bytes) for one cache length and batch."""
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
    nbytes = cache_bytes(cache)
    cur = torch.randint(1000, 30000, (batch, block), generator=gen).to(dev)
    mask = torch.ones(1, block, length + block, device=dev)
    pos = torch.arange(length, length + block, device=dev).unsqueeze(0).expand(batch, -1)

    def fwd(store):
        model(cur, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True, store_kv=store)
        if store:
            cache.crop(length)                       # keep the cache at `length` for the next rep

    def timed(store, n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fwd(store)
        torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - t0) / n

    for _ in range(warmup):
        fwd(False)
    ms_step = timed(False, reps)
    ms_commit = timed(True, max(reps // 4, 3))
    del cache
    return ms_step, ms_commit, nbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sdar8b", choices=["sdar8b", "sdar30b"])
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3, help="interleaved rounds; the minimum is kept")
    ap.add_argument("--bits", type=float, default=3.0, help="honest bits per number of the real compressed cache")
    ap.add_argument("--bandwidth", type=float, default=4.8, help="GPU memory bandwidth, TB/s (H200: 4.8)")
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
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    util_start = gpu_util()
    points = [(b, n) for b in args.batch for n in args.lengths]
    meas = {p: [] for p in points}
    skipped = set()
    for rnd in range(args.rounds):
        for batch, length in points:
            if (batch, length) in skipped:
                continue
            try:
                ms, mc, nbytes = time_point(model, length, args.block, batch, args.reps, args.warmup)
            except torch.OutOfMemoryError:
                print(f"length {length} batch {batch}: out of memory, skipped", flush=True)
                skipped.add((batch, length))
                torch.cuda.empty_cache()
                continue
            meas[(batch, length)].append((ms, mc, nbytes))
            print(f"round {rnd + 1} block {args.block} length {length} batch {batch}: "
                  f"{ms:.2f} ms/step, {mc:.2f} ms/commit, cache {nbytes / 2**30:.2f} GiB", flush=True)
            torch.cuda.empty_cache()
    util_end = gpu_util()

    rows = []
    for (batch, length), ms in meas.items():
        if not ms:
            continue
        steps, commits = [m[0] for m in ms], [m[1] for m in ms]
        rows.append(dict(model=args.model, block=args.block, length=length, batch=batch,
                         ms_step=round(min(steps), 3), ms_step_max=round(max(steps), 3),
                         ms_commit=round(min(commits), 3), rounds=len(ms), cache_bytes=ms[0][2]))
    rows = cost_model(rows, args.bits, weight_bytes, args.bandwidth)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(vars(args), repo=repo_id, weight_bytes=weight_bytes,
                gpu_util_start=util_start, gpu_util_end=util_end, **device_info())
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"GPU utilization before / after: {util_start} / {util_end} %; weights {weight_bytes / 2**30:.1f} GiB")
    print(f"{'len':>6s} {'batch':>5s} {'ms/step':>8s} {'max':>7s} {'commit':>7s} {'cacheGiB':>8s} {'ms/GiB':>7s} "
          f"{'2bit ms':>8s} {'b-even':>7s} {'ideal16':>8s} {'ideal2':>7s} {'ideal b-even':>12s}")
    for r in rows:
        print(f"{r['length']:6d} {r['batch']:5d} {r['ms_step']:8.2f} {r['ms_step_max']:7.2f} {r['ms_commit']:7.2f} "
              f"{r['cache_gb']:8.2f} {r['ms_per_gb']:7.2f} {r['est_ms_2bit']:8.2f} {100 * r['break_even']:6.1f}% "
              f"{r['ideal_ms_16']:8.2f} {r['ideal_ms_2bit']:7.2f} {100 * r['ideal_break_even']:11.1f}%")


if __name__ == "__main__":
    main()
