"""SDAR or Fast-dLLM v2 on LongProc (long input, long structured output: 0.5K / 2K / 8K tokens),
with or without KV quantization. LongProc's own prompts and scoring are used: its repository
(github.com/princeton-pli/LongProc, `longproc/` and `data/`) must be on disk, at --longproc.

    python -m blockkv.run_longproc --model sdar8b --block 32 --longproc ~/third_party/LongProc \
        --datasets path_traversal_8k tom_tracking_8k --per-task 50 --kv-bits 2 \
        --out out/longproc/8k/sdar8b_b32_kv2.jsonl

Each line: idx ("<dataset>:<i>"), task, score (the task's main metric: f1 for html_to_tsv, else
partial accuracy where the task has it, else accuracy), all metrics, gen_tokens, prompt_tokens, steps, per_step,
text. Readable by blockkv.summarize_ruler and blockkv.tax_decompose (--cap = the answer limit).
Answer limit as in LongProc's example_usage.py: 1024 / 3072 / 9216 tokens for 0.5K / 2K / 8K.
Problems whose prompt is longer than --max-input tokens are skipped (and counted in the meta file):
cutting a procedure's input would change the task.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

from blockkv import decode
from blockkv.kvquant import KVQuant
from blockkv.run_gsm8k import device_info, mask_id, patch_rmsnorm, read_done, repo, resolve_path, stop_ids

MAX_NEW = {"0.5k": 1024, "2k": 3072, "8k": 9216}
MAIN = ("f1", "partial_accuracy", "accuracy")   # first metric present is the score (partial credit for
                                                # long procedures, where an 8B model rarely gets all of it)


def answer_limit(dataset):
    return MAX_NEW[dataset.rsplit("_", 1)[1]]


def main_score(metrics):
    for k in MAIN:
        if k in metrics:
            return float(metrics[k])
    return float(next(iter(metrics.values())))


def load(longproc, datasets, per_task, seed):
    """[(idx, task, prompt, max_new, eval_fn, example)], a fixed random subset per dataset."""
    root = Path(longproc).expanduser()
    sys.path.insert(0, str(root))
    from longproc.longproc_data import load_longproc_data
    out = []
    for ds in datasets:
        data, ev = load_longproc_data(ds, str(root / "data"))
        idx = list(range(len(data)))
        if per_task and per_task < len(idx):
            idx = sorted(random.Random(seed).sample(idx, per_task))
        out += [(f"{ds}:{i}", ds, data[i]["input_prompt"], answer_limit(ds), ev, data[i]) for i in idx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sdar8b", choices=["sdar8b", "sdar30b", "fastdllm"])
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--small-block", type=int, default=8, help="Fast-dLLM v2 sub-block")
    ap.add_argument("--longproc", required=True, help="the LongProc repository folder (with longproc/ and data/)")
    ap.add_argument("--datasets", nargs="+", default=["path_traversal_8k", "html_to_tsv_8k", "tom_tracking_8k",
                                                       "countdown_8k", "travel_planning_8k"])
    ap.add_argument("--per-task", type=int, default=50, help="problems per dataset (fixed random subset; 0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-input", type=int, default=24576, help="skip problems with a longer prompt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--prefill-chunk", type=int, default=1024)
    ap.add_argument("--kv-bits", type=int, choices=[2, 3, 4, 8, 16])
    ap.add_argument("--kv-only", choices=["k", "v"])
    ap.add_argument("--kv-group", type=int, default=32)
    ap.add_argument("--kv-window", type=int, default=0)
    ap.add_argument("--kv-sink", type=int, default=0)
    ap.add_argument("--key-mode", choices=["channel", "token"], default="channel")
    ap.add_argument("--kv-rotate", action="store_true")
    ap.add_argument("--kv-k-bits", type=int, choices=[2, 3, 4, 8, 16])
    ap.add_argument("--kv-v-bits", type=int, choices=[2, 3, 4, 8, 16])
    ap.add_argument("--kv-key-boost", type=float, default=0.0)
    ap.add_argument("--kv-key-boost-bits", type=int, default=8)
    ap.add_argument("--kv-rotate-values", action="store_true")
    ap.add_argument("--torch-rmsnorm", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    family, repo_id = repo(args.model, args.block)
    path = resolve_path(repo_id, args.model_path)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16 if family == "sdar" else torch.bfloat16, trust_remote_code=True,
        device_map="cuda", low_cpu_mem_usage=True).eval()
    mid, stops = mask_id(family, tok, None), stop_ids(family, tok, model)
    rmsnorm_patched = patch_rmsnorm(model) if args.torch_rmsnorm else False
    if args.torch_rmsnorm and not rmsnorm_patched:
        raise SystemExit("--torch-rmsnorm: no flash_rms_norm in this model's code")

    work = load(args.longproc, args.datasets, args.per_task, args.seed)[:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(out)
    skipped = []
    meta = dict(vars(args), repo=repo_id, mask_id=mid, stop_ids=stops, rmsnorm_patched=rmsnorm_patched,
                n_items=len(work), **device_info())
    with open(out, "a", encoding="utf-8") as f:
        for n, (i, task, prompt, limit, ev, example) in enumerate(work):
            if i in done:
                continue
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True)
            ids = tok(text, return_tensors="pt", add_special_tokens=family != "sdar").input_ids
            if ids.shape[1] > args.max_input:
                skipped.append(i)
                print(f"[{n + 1}/{len(work)}] {i} skipped: prompt {ids.shape[1]} > {args.max_input}", flush=True)
                continue
            ids = ids.cuda()
            quant = (KVQuant(args.kv_bits, args.kv_group, args.kv_window, args.kv_sink, args.key_mode, args.kv_only,
                             rotate="v" if args.kv_rotate_values else args.kv_rotate,
                             k_bits=args.kv_k_bits, v_bits=args.kv_v_bits,
                             key_boost=args.kv_key_boost, key_boost_bits=args.kv_key_boost_bits)
                     if args.kv_bits else None)
            t0 = time.time()
            if family == "sdar":
                gen, st = decode.sdar_generate(model, ids, mid, args.block, args.block, args.threshold, limit, stops,
                                               on_store=quant, prefill_chunk=args.prefill_chunk or None)
            else:
                gen, st = decode.fastdllm_generate(model, ids, mid, args.block, args.small_block, args.threshold,
                                                   limit, stops, on_store=quant,
                                                   prefill_chunk=args.prefill_chunk or None)
            gen = gen[:limit]
            answer = tok.decode(gen, skip_special_tokens=True)
            try:
                metrics, _ = ev(answer, example)
            except Exception as e:                       # a scorer crash scores 0, the run goes on
                metrics = {"accuracy": 0.0, "scorer_error": str(e)[:200]}
            s = main_score({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            row = dict(idx=i, task=task, score=s, correct=s, metrics=metrics, gen_tokens=len(gen),
                       prompt_tokens=ids.shape[1], steps=st.steps, nfe=st.nfe, forced=st.forced, blocks=st.blocks,
                       per_step=st.per_step, sec=round(time.time() - t0, 2), text=answer)
            if quant is not None:
                row.update(quant.summary(st.cache_len))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n + 1}/{len(work)}] {i} score={s:.3f} tokens={row['gen_tokens']} prompt={row['prompt_tokens']} "
                  f"steps={row['steps']} {row['sec']}s", flush=True)
    meta["skipped"] = skipped
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
