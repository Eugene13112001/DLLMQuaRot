"""SDAR on RULER (long inputs), with or without KV quantization.

    python -m blockkv.run_ruler --block 16 --length 16384 --data ~/data/ruler \
        --kv-bits 2 --out out/ruler/16384/sdar8b_b16_kv2.jsonl

Data: the `simonjegou/ruler` parquet files, one folder per input length. The prompt
and the scoring follow NVIDIA's KVPress evaluation: user message = context + question,
then the answer prefix; `string_match_part` for the qa tasks, `string_match_all` for
the others. Answers are cut at the task's `max_new_tokens`: the decoder fills whole
blocks, so without the cut a larger block would get extra tokens to match against.
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import torch

from blockkv import decode
from blockkv.kvquant import KVQuant
from blockkv.run_gsm8k import mask_id, read_done, repo, resolve_path, stop_ids

CONTROL = re.compile(r"[\x00-\x1f]")


def score(task, pred, answers):
    """1 point per sample at most: any answer for qa tasks, the share of answers otherwise."""
    pred = CONTROL.sub("", pred.strip()).strip().lower()
    hits = [a.lower() in pred for a in answers]
    return float(any(hits)) if task.split("_")[0] == "qa" else sum(hits) / len(hits)


def load(root, length):
    """Rows of one input length, as plain dicts."""
    import pandas as pd
    files = sorted((Path(root).expanduser() / str(length)).rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {root}/{length}")
    rows = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).to_dict("records")
    return [dict(r, answer=[str(a) for a in r["answer"]]) for r in rows]


def pick(rows, per_task, seed):
    """Row indices: `per_task` random rows of every task, always the same ones."""
    if not per_task:
        return list(range(len(rows)))
    tasks = {}
    for i, r in enumerate(rows):
        tasks.setdefault(r["task"], []).append(i)
    rng = random.Random(seed)
    return sorted(i for t in sorted(tasks) for i in rng.sample(tasks[t], min(per_task, len(tasks[t]))))


def prompt_text(tok, row):
    return tok.apply_chat_template([{"role": "user", "content": row["context"] + row["question"]}],
                                   tokenize=False, add_generation_prompt=True) + row["answer_prefix"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sdar8b", choices=["sdar8b", "sdar30b"])
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--length", type=int, required=True, help="input length: a folder of the dataset (4096, 8192, 16384)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path")
    ap.add_argument("--per-task", type=int, default=30, help="problems per task, 0 for all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--prefill-chunk", type=int, default=1024,
                    help="feed the prompt in pieces of this many tokens (0: one forward, needs tens of GiB at 16K)")
    ap.add_argument("--kv-bits", type=int, choices=[2, 3, 4, 8, 16])
    ap.add_argument("--kv-only", choices=["k", "v"])
    ap.add_argument("--kv-group", type=int, default=32)
    ap.add_argument("--kv-window", type=int, default=0)
    ap.add_argument("--kv-sink", type=int, default=0)
    ap.add_argument("--key-mode", choices=["channel", "token"], default="channel")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    family, repo_id = repo(args.model, args.block)
    path = resolve_path(repo_id, args.model_path)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda", low_cpu_mem_usage=True).eval()
    mid, stops = mask_id(family, tok, None), stop_ids(family, tok, model)

    rows = load(args.data, args.length)
    idxs = pick(rows, args.per_task, args.seed)[:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(out)
    out.with_suffix(".meta.json").write_text(json.dumps(dict(vars(args), repo=repo_id, mask_id=mid, stop_ids=stops), indent=1),
                                             encoding="utf-8")

    with open(out, "a", encoding="utf-8") as f:
        for n, i in enumerate(idxs):
            if i in done:
                continue
            r = rows[i]
            ids = tok(prompt_text(tok, r), return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            limit = int(r["max_new_tokens"])
            quant = (KVQuant(args.kv_bits, args.kv_group, args.kv_window, args.kv_sink, args.key_mode, args.kv_only)
                     if args.kv_bits else None)
            t0 = time.time()
            gen, st = decode.sdar_generate(model, ids, mid, args.block, args.block, args.threshold, limit, stops,
                                           on_store=quant, prefill_chunk=args.prefill_chunk or None)
            gen = gen[:limit]
            answer = tok.decode(gen, skip_special_tokens=True)
            s = score(r["task"], answer, r["answer"])
            row = dict(idx=i, task=r["task"], score=s, correct=s == 1.0, gen_tokens=len(gen), prompt_tokens=ids.shape[1],
                       steps=st.steps, nfe=st.nfe, forced=st.forced, blocks=st.blocks, per_step=st.per_step,
                       sec=round(time.time() - t0, 2), text=answer)
            if quant is not None:
                row.update(quant.summary(st.cache_len))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n + 1}/{len(idxs)}] idx={i} {r['task']} score={s:.2f} tokens={row['gen_tokens']} "
                  f"prompt={row['prompt_tokens']} steps={row['steps']} {row['sec']}s", flush=True)


if __name__ == "__main__":
    main()
