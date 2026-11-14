"""Real 2-bit step time on Fast-dLLM v2 with BitSieve's kernels, measured cleanly on a
shared server.

    nohup python -m blockkv.bitsieve_timing --bitsieve ~/third_party/BitSieve \
        --out out/timing/bitsieve > out/timing/bitsieve/orchestrator.log 2>&1 &

1. Waits for an idle card among --cards (nvidia-smi utilization <= --max-util and at
   least --need-gb free, --checks checks in a row, one check every --sleep seconds).
2. Measures every (batch, context) point with every config in turn (16-bit then 2-bit),
   so a change on the card affects both configs of a point alike. Each run is BitSieve's
   own `bitsieve_fastdllm.eval.performance` for one config and one point.
3. Before each run waits until the card is idle again (nobody computing on it).
4. During each run samples the card every --poll seconds: memory used by anyone but our
   run, and processes that are not ours. A run during which foreign memory grew by more
   than --foreign-gb, or a foreign process appeared, is marked dirty and repeated, up to
   --retries times. nvidia-smi on this server misses processes of other containers, so
   the memory is the main check; a neighbour that only starts computing with memory it
   already held is caught by the idle check before the next run, not during this one.

Output in --out: `runs.jsonl` (one line per run: card, attempt, clean, foreign memory,
BitSieve's per-repeat rows), `summary.jsonl` (one line per point: median time per block
for each config, speedup), and the raw BitSieve files under `raw/`.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

CONFIGS = ["official_dense_bf16", "dense_kivi2_k2v2_r32"]


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


def smi(query, kind="gpu"):
    """Rows of nvidia-smi CSV output (no header, no units)."""
    flag = "--query-gpu" if kind == "gpu" else "--query-compute-apps"
    out = subprocess.run(["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout
    return [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]


def card_uuid(card):
    cmd = [sys.executable, "-c", "import torch; print(str(torch.cuda.get_device_properties(0).uuid))"]
    out = subprocess.run(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(card)),
                         capture_output=True, text=True, timeout=120).stdout.strip()
    return out.replace("GPU-", "")


def free_gb(card):
    cmd = [sys.executable, "-c", "import torch; print(torch.cuda.mem_get_info(0)[0] / 2**30)"]
    out = subprocess.run(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(card)),
                         capture_output=True, text=True, timeout=120).stdout.strip()
    return float(out or 0)


def gpu_state(uuid):
    """(utilization %, memory used MiB) of one GPU."""
    for u, util, mem in smi("uuid,utilization.gpu,memory.used"):
        if u.replace("GPU-", "").startswith(uuid[:8]):
            return int(util), int(mem)
    return 100, 0


def apps(uuid):
    """{pid: used MiB} of the compute processes nvidia-smi sees on one GPU."""
    out = {}
    for g, pid, mem in smi("gpu_uuid,pid,used_memory", kind="apps"):
        if g.replace("GPU-", "").startswith(uuid[:8]) and pid.isdigit():
            out[int(pid)] = int(mem) if mem.isdigit() else 0
    return out


def child_pids(pid):
    """pid and all its descendants (from /proc)."""
    pids, todo = {pid}, [pid]
    while todo:
        p = todo.pop()
        try:
            kids = Path(f"/proc/{p}/task/{p}/children").read_text().split()
        except OSError:
            kids = []
        for k in map(int, kids):
            if k not in pids:
                pids.add(k)
                todo.append(k)
    return pids


def wait_idle(cards, uuids, max_util, need_gb, checks, sleep, only=None):
    """First card that is idle `checks` times in a row (or wait for `only`)."""
    count = {c: 0 for c in cards}
    watch = [only] if only is not None else cards
    while True:
        for c in watch:
            util, mem = gpu_state(uuids[c])
            free = free_gb(c) if util <= max_util else -1
            count[c] = count[c] + 1 if util <= max_util and free >= need_gb else 0
            log(f"card {c}: util {util}% used {mem} MiB free {free:.0f} GiB idle {count[c]}/{checks}")
            if count[c] >= checks:
                return c
        time.sleep(sleep)


class Watch(threading.Thread):
    """Samples one GPU while our run is alive: foreign memory and foreign processes."""

    def __init__(self, uuid, root_pid, poll):
        super().__init__(daemon=True)
        self.uuid, self.root, self.poll = uuid, root_pid, poll
        self.stop = threading.Event()
        self.base = None
        self.max_extra_mib, self.foreign_pids, self.samples, self.utils = 0, set(), 0, []

    def run(self):
        while not self.stop.is_set():
            try:
                util, used = gpu_state(self.uuid)
                ours = child_pids(self.root)
                a = apps(self.uuid)
                our_mem = sum(m for p, m in a.items() if p in ours)
                foreign = used - our_mem
                if self.base is None:
                    self.base = foreign
                self.max_extra_mib = max(self.max_extra_mib, foreign - self.base)
                self.foreign_pids |= {p for p in a if p not in ours}
                self.utils.append(util)
                self.samples += 1
            except Exception as e:                       # a failed sample must not kill the run
                log("watch error:", e)
            self.stop.wait(self.poll)


def cache_tokens(context, new_tokens, block=32):
    """Cache capacity for one point: context + answer, rounded up to whole blocks."""
    return -(-(context + new_tokens) // block) * block


def run_point(args, card, uuid, cfg, batch, context, attempt, base_pids):
    raw = Path(args.out) / "raw" / f"{cfg}_ctx{context}_b{batch}_a{attempt}.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "bitsieve_fastdllm.eval.performance",
           "--config", str(Path(args.bitsieve) / "configs" / f"{cfg}.yaml"),
           "--contexts", str(context), "--batch-sizes", str(batch),
           "--max-new-tokens", str(args.max_new_tokens),
           "--max-cache-tokens", str(cache_tokens(context, args.max_new_tokens)),
           "--schedule", "fixed", "--fixed-steps", str(args.fixed_steps),
           "--warmup", str(args.warmup), "--repeats", str(args.repeats), "--output", str(raw)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(card), HF_HUB_OFFLINE="1",
               PYTHONPATH=str(Path(args.bitsieve) / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    with open(raw.with_suffix(".log"), "w", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=args.bitsieve)
        w = Watch(uuid, p.pid, args.poll)
        w.start()
        t0 = time.time()
        rc = p.wait()
        w.stop.set()
        w.join()
    rows = [json.loads(l) for l in open(raw, encoding="utf-8") if l.strip()] if raw.exists() else []
    new_foreign = sorted(w.foreign_pids - base_pids)
    clean = rc == 0 and bool(rows) and w.max_extra_mib <= args.foreign_gb * 1024 and not new_foreign
    return dict(card=card, uuid=uuid[:8], config=cfg, batch=batch, context=context, attempt=attempt,
                rc=rc, seconds=round(time.time() - t0, 1), clean=clean,
                foreign_extra_mib=w.max_extra_mib, new_foreign_pids=new_foreign,
                util_samples=w.samples, util_median=statistics.median(w.utils) if w.utils else None,
                rows=rows)


def median_tpob(rows):
    vals = [float(r["mean_tpob_ms"]) for r in rows if r.get("mean_tpob_ms") is not None]
    return statistics.median(vals) if vals else None


def summarize(runs, configs=CONFIGS):
    """One line per (batch, context): median time per block of the last clean run of each config."""
    best = {}
    for r in runs:
        key = (r["batch"], r["context"], r["config"])
        if r["clean"] or key not in best or not best[key]["clean"]:
            best[key] = r
    out = []
    for batch, context in sorted({(b, c) for b, c, _ in best}):
        row = dict(batch=batch, context=context)
        for cfg in configs:
            r = best.get((batch, context, cfg))
            row[f"{cfg}_tpob_ms"] = round(median_tpob(r["rows"]), 3) if r and median_tpob(r["rows"]) else None
            row[f"{cfg}_clean"] = bool(r and r["clean"])
        a, b = row.get(f"{configs[0]}_tpob_ms"), row.get(f"{configs[1]}_tpob_ms")
        row["speedup"] = round(a / b, 4) if a and b else None
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitsieve", default=str(Path("~/third_party/BitSieve").expanduser()))
    ap.add_argument("--out", default="out/timing/bitsieve")
    ap.add_argument("--cards", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--contexts", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--fixed-steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-util", type=int, default=5)
    ap.add_argument("--need-gb", type=float, default=60)
    ap.add_argument("--checks", type=int, default=3)
    ap.add_argument("--sleep", type=int, default=60)
    ap.add_argument("--poll", type=int, default=5)
    ap.add_argument("--foreign-gb", type=float, default=1.0)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()
    args.bitsieve = str(Path(args.bitsieve).expanduser())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    uuids = {c: card_uuid(c) for c in args.cards}
    for c, u in uuids.items():
        log(f"card {c} = GPU-{u}")
    card = wait_idle(args.cards, uuids, args.max_util, args.need_gb, args.checks, args.sleep)
    uuid = uuids[card]
    base_pids = set(apps(uuid))
    log(f"card {card} is idle; visible processes already there: {sorted(base_pids)}")
    (out / "where.txt").write_text(f"card {card} GPU-{uuid}, started {time.strftime('%F %T')}\n", encoding="utf-8")

    runs = []
    for batch in args.batches:
        for context in args.contexts:
            for cfg in args.configs:
                for attempt in range(args.retries + 1):
                    wait_idle(args.cards, uuids, args.max_util, args.need_gb, 1, 30, only=card)
                    r = run_point(args, card, uuid, cfg, batch, context, attempt, base_pids)
                    runs.append(r)
                    with open(out / "runs.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(r) + "\n")
                    log(f"{cfg} batch {batch} ctx {context} attempt {attempt}: rc {r['rc']}, "
                        f"tpob {median_tpob(r['rows'])}, clean {r['clean']}, foreign +{r['foreign_extra_mib']} MiB, "
                        f"new foreign pids {r['new_foreign_pids']}")
                    if r["clean"] or r["rc"] != 0 and not r["rows"]:
                        break                            # clean, or a hard failure (e.g. OOM): no retry

    summary = summarize(runs, args.configs)
    with open(out / "summary.jsonl", "w", encoding="utf-8") as f:
        for s in summary:
            f.write(json.dumps(s) + "\n")
    with open(out / "where.txt", "a", encoding="utf-8") as f:
        f.write(f"done {time.strftime('%F %T')}\n")
    log("summary:")
    for s in summary:
        log(json.dumps(s))


if __name__ == "__main__":
    main()
