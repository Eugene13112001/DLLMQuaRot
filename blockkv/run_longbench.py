"""SDAR or Fast-dLLM v2 on LongBench qmsum + gov_report (long input, long answer), with or
without KV quantization; one JSON line per problem, readable by blockkv.summarize_ruler.

    python -m blockkv.run_longbench --model sdar8b --block 16 --data ~/data/longbench/data.zip \
        --max-input 16384 --kv-bits 2 --out out/longbench/16384/sdar8b_b16_kv2.jsonl

Same decoders, quantizer and chunked prefill as blockkv.run_ruler.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from blockkv import decode, longbench
from blockkv.kvquant import KVQuant
from blockkv.run_gsm8k import device_info, mask_id, patch_rmsnorm, read_done, repo, resolve_path, stop_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sdar8b", choices=["sdar8b", "sdar30b", "fastdllm"])
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--small-block", type=int, default=8, help="Fast-dLLM v2 sub-block")
    ap.add_argument("--data", required=True, help="LongBench data.zip or a folder of <task>.jsonl")
    ap.add_argument("--tasks", nargs="+", default=["qmsum", "gov_report"])
    ap.add_argument("--max-input", type=int, default=16384, help="prompt cut in the middle to this many tokens")
    ap.add_argument("--per-task", type=int, help="first N problems of each task (default all: 200 + 200)")
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

    work = longbench.items(args.tasks, args.data, tok, args.max_input, args.per_task)[:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(out)
    out.with_suffix(".meta.json").write_text(json.dumps(dict(vars(args), repo=repo_id, mask_id=mid, stop_ids=stops,
                                                                  rmsnorm_patched=rmsnorm_patched, **device_info()),
                                                             indent=1), encoding="utf-8")
    with open(out, "a", encoding="utf-8") as f:
        for n, (i, task, r) in enumerate(work):
            if i in done:
                continue
            text = longbench.chat_prompt(task, r, tok, args.max_input)
            ids = tok(text, return_tensors="pt", add_special_tokens=family != "sdar").input_ids.cuda()
            limit = longbench.MAX_NEW[task]
            quant = (KVQuant(args.kv_bits, args.kv_group, args.kv_window, args.kv_sink, args.key_mode, args.kv_only)
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
            s = longbench.score(answer, [str(a) for a in r["answers"]])
            row = dict(idx=i, task=task, score=s, correct=s, gen_tokens=len(gen), prompt_tokens=ids.shape[1],
                       steps=st.steps, nfe=st.nfe, forced=st.forced, blocks=st.blocks, per_step=st.per_step,
                       sec=round(time.time() - t0, 2), text=answer)
            if quant is not None:
                row.update(quant.summary(st.cache_len))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n + 1}/{len(work)}] {i} rougeL={s:.3f} tokens={row['gen_tokens']} prompt={row['prompt_tokens']} "
                  f"steps={row['steps']} {row['sec']}s", flush=True)


if __name__ == "__main__":
    main()
