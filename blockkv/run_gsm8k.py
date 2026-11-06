"""Run one model configuration on GSM8K and write one JSON line per problem.

    python -m blockkv.run_gsm8k --model sdar8b --block 16 --out out/sdar8b_b16.jsonl

Each line: idx, correct, pred, gold, gen_tokens, steps, nfe, forced, blocks,
per_step, text. An existing output file is resumed: finished idx are skipped.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

from blockkv import decode, gsm8k

def repo(model, block):
    """(family, HF repo id). SDAR's unsuffixed release is the block-4 model."""
    if model in ("sdar8b", "sdar30b"):
        base = {"sdar8b": "JetLM/SDAR-8B-Chat", "sdar30b": "JetLM/SDAR-30B-A3B-Chat"}[model]
        return "sdar", base if block == 4 else f"{base}-b{block}"
    if model == "fastdllm":
        assert block == 32, "Fast-dLLM v2 is trained at block 32 only"
        return "fastdllm", "Efficient-Large-Model/Fast_dLLM_v2_7B"
    raise ValueError(model)


def resolve_path(repo_id, override):
    if override:
        return override
    root = os.environ.get("MODELS_DIR")
    if root and (Path(root) / repo_id.split("/")[-1]).exists():
        return str(Path(root) / repo_id.split("/")[-1])
    return repo_id


# Per family, as in the authors' GSM8K evaluation:
#   sdar:     OpenCompass huggingface_bd3.py — float16, gen_length 4096, stops = generation_config eos
#             + tokenizer eos, templated prompt encoded without special tokens.
#   fastdllm: Fast-dLLM v2/eval.py — bfloat16, max_new_tokens 2048, generate() stops on 151645 only.
DEFAULTS = {
    "sdar": dict(dtype="float16", gen_length=4096),
    "fastdllm": dict(dtype="bfloat16", gen_length=2048),
}


def stop_ids(family, tok, model):
    if family == "fastdllm":
        return [tok.convert_tokens_to_ids("<|im_end|>")]
    ids = set()
    for e in (tok.eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
        ids.update(e if isinstance(e, (list, tuple)) else [e] if e is not None else [])
    return sorted(ids)


def read_done(out):
    """idx already in `out`. A run killed mid-write leaves a cut last line: drop it."""
    if not out.exists():
        return set()
    good, done = [], set()
    with open(out, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["idx"])
                good.append(line if line.endswith("\n") else line + "\n")
            except (json.JSONDecodeError, KeyError):
                print(f"dropping a broken line in {out}", flush=True)
    out.write_text("".join(good), encoding="utf-8")
    return done


def mask_id(family, tok, override):
    if override is not None:
        return override
    if family == "sdar":
        if tok.mask_token_id is not None:
            return tok.mask_token_id
        i = tok.convert_tokens_to_ids("<|MASK|>")
        return i if isinstance(i, int) and i != tok.unk_token_id else 151669
    return 151665  # Fast-dLLM v2 generate() default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["sdar8b", "sdar30b", "fastdllm"])
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--model-path", help="local checkpoint dir (default: $MODELS_DIR/<repo name>)")
    ap.add_argument("--data", help="GSM8K test .jsonl or .parquet (default: HF openai/gsm8k)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="first N problems only")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--gen-length", type=int, help="default: 4096 SDAR, 2048 Fast-dLLM v2")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="as in both authors' GSM8K evaluation: 1.0, one token per step")
    ap.add_argument("--steps", type=int, help="SDAR denoising steps per block (default: block size)")
    ap.add_argument("--small-block", type=int, default=8, help="Fast-dLLM v2 sub-block")
    ap.add_argument("--prompt-style", choices=list(gsm8k.PROMPT_STYLES),
                    help="default: the model family's own evaluation prompt")
    ap.add_argument("--mask-id", type=int)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], help="default: float16 SDAR, bfloat16 Fast-dLLM v2")
    ap.add_argument("--check-native", type=int, default=0,
                    help="Fast-dLLM v2: also run model.generate() on the first N problems and assert identical tokens")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    family, repo_id = repo(args.model, args.block)
    path = resolve_path(repo_id, args.model_path)
    threshold = args.threshold
    steps = args.steps or args.block
    style = args.prompt_style or family
    gen_length = args.gen_length or DEFAULTS[family]["gen_length"]
    dtype = args.dtype or DEFAULTS[family]["dtype"]

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    # Straight onto the GPU, shard by shard: loading on the CPU first needs the whole
    # model in host RAM per process, and several runs at once get OOM-killed.
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=getattr(torch, dtype), trust_remote_code=True,
        device_map="cuda", low_cpu_mem_usage=True).eval()
    mid, stops = mask_id(family, tok, args.mask_id), stop_ids(family, tok, model)

    problems = gsm8k.load(args.data)
    idxs = list(range(len(problems)))[:args.limit][args.shard::args.num_shards]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(out)
    meta = dict(vars(args), family=family, repo=repo_id, path=path, threshold=threshold,
                steps=steps, prompt_style=style, gen_length=gen_length, dtype=dtype,
                mask_id=mid, stop_ids=stops)
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta), flush=True)

    with open(out, "a", encoding="utf-8") as f:
        for n, i in enumerate(idxs):
            if i in done:
                continue
            p = problems[i]
            # No system message: Fast-dLLM v2's template inserts its default one itself, SDAR's adds none.
            text = tok.apply_chat_template(
                [{"role": "user", "content": gsm8k.PROMPT_STYLES[style].format(question=p["question"])}],
                tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt", add_special_tokens=family != "sdar").input_ids.cuda()
            t0 = time.time()
            if family == "sdar":
                gen, st = decode.sdar_generate(model, ids, mid, args.block, steps, threshold,
                                               gen_length, stops)
            else:
                gen, st = decode.fastdllm_generate(model, ids, mid, args.block, args.small_block,
                                                   threshold, gen_length, stops)
                if n < args.check_native:
                    ref = model.generate(ids, tokenizer=tok, max_new_tokens=gen_length, mask_id=mid,
                                         threshold=threshold, small_block_size=args.small_block,
                                         block_size=args.block, stop_token=stops[0],
                                         temperature=0)[0, ids.shape[1]:]
                    s = decode.first_stop(ref, stops)
                    ref = ref if s is None else ref[:s + 1]
                    assert torch.equal(ref, gen), f"problem {i}: our loop differs from model.generate()"
            answer = tok.decode(gen, skip_special_tokens=True)
            pred, ref_ans = gsm8k.extract(answer), gsm8k.gold(p["answer"])
            row = dict(idx=i, correct=gsm8k.is_correct(pred, ref_ans), pred=pred, gold=ref_ans,
                       gen_tokens=len(gen), steps=st.steps, nfe=st.nfe, forced=st.forced,
                       blocks=st.blocks, per_step=st.per_step, sec=round(time.time() - t0, 2),
                       text=answer)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n + 1}/{len(idxs)}] idx={i} correct={row['correct']} tokens={row['gen_tokens']} "
                  f"steps={row['steps']} {row['sec']}s", flush=True)


if __name__ == "__main__":
    main()
