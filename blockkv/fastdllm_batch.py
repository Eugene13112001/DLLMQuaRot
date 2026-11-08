"""Fast-dLLM v2 through the authors' batched sampler, with or without KV quantization.

SoloQ and the NVlabs harness evaluate GSM8K at batch size 32 with
generation_functions.batch_sample; run_gsm8k.py decodes one problem at a time.
This script checks whether that difference explains the gap to SoloQ's numbers.

    python -m blockkv.fastdllm_batch --gen-fn ~/data/generation_functions.py \
        --data ~/data/gsm8k_test.jsonl --out out/batch/fp.jsonl
"""
import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

from blockkv import decode, gsm8k
from blockkv.kvquant import KVQuant

REPO, MASK, STOP = "Efficient-Large-Model/Fast_dLLM_v2_7B", 151665, 151645


def make_batches(lengths, size):
    """Indices sorted by prompt length and cut into batches, as the NVlabs harness does."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[i:i + size] for i in range(0, len(order), size)]


def pad_right(prompts, mask_id, device):
    """(batch, seq_len): prompts padded on the right with the mask token."""
    n = max(len(p) for p in prompts)
    rows = [p + [mask_id] * (n - len(p)) for p in prompts]
    return (torch.tensor(rows, dtype=torch.long, device=device),
            torch.tensor([len(p) for p in prompts], device=device))


def hook_store(model, forward, quant):
    """Quantize the cache after every forward that stores a committed block."""
    def wrapped(*args, **kw):
        out = forward(*args, **kw)
        if kw.get("update_past_key_values"):
            n = out.past_key_values.get_seq_length()
            quant(out.past_key_values, n - kw["input_ids"].shape[1], n)
        return out
    model.forward = wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-fn", required=True, help="NVlabs Fast-dLLM v2/generation_functions.py")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--kv-bits", type=int, choices=[2, 3, 4, 8, 16])
    ap.add_argument("--kv-only", choices=["k", "v"])
    ap.add_argument("--shared-scale", action="store_true", help="one (min, step) for the whole batch")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = importlib.util.spec_from_file_location("generation_functions", args.gen_fn)
    gen_fn = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gen_fn     # auto_docstring looks the class's file up there
    spec.loader.exec_module(gen_fn)

    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        REPO, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="cuda", low_cpu_mem_usage=True).eval()
    model.batch_sample = types.MethodType(gen_fn.Fast_dLLM_QwenForCausalLM.batch_sample, model)

    problems = gsm8k.load(args.data)[:args.limit]
    prompts = []
    for p in problems:
        text = tok.apply_chat_template(
            [{"role": "user", "content": gsm8k.PROMPT_STYLES["fastdllm"].format(question=p["question"])}],
            tokenize=False, add_generation_prompt=True)
        prompts.append(tok(text).input_ids)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".meta.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")
    plain_forward = model.forward
    batches = make_batches([len(p) for p in prompts], args.batch_size)
    with open(out, "w", encoding="utf-8") as f, torch.no_grad():
        for b, idxs in enumerate(batches):
            ids, seq_len = pad_right([prompts[i] for i in idxs], MASK, "cuda")
            quant = None
            model.forward = plain_forward
            if args.kv_bits:
                quant = KVQuant(args.kv_bits, 32, only=args.kv_only, shared=args.shared_scale)
                hook_store(model, plain_forward, quant)
            done = model.batch_sample(ids, tokenizer=tok, block_size=32, small_block_size=8,
                                      max_new_tokens=args.max_new_tokens, mask_id=MASK,
                                      min_len=int(seq_len.min()), seq_len=seq_len.clone(),
                                      threshold=args.threshold, stop_token=STOP)
            n_right = 0
            for pos, i in enumerate(idxs):
                gen = done[pos][int(seq_len[pos]):]
                s = decode.first_stop(gen, [STOP])
                gen = gen if s is None else gen[:s + 1]
                answer = tok.decode(gen, skip_special_tokens=True)
                pred, ref = gsm8k.extract(answer), gsm8k.gold(problems[i]["answer"])
                row = dict(idx=i, correct=gsm8k.is_correct(pred, ref), pred=pred, gold=ref,
                           gen_tokens=len(gen), steps=0, nfe=0, forced=0, blocks=0, per_step=[],
                           text=answer)
                if quant is not None:
                    row.update(quant.summary(quant.done))
                n_right += row["correct"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{b + 1}/{len(batches)}] correct {n_right}/{len(idxs)}", flush=True)


if __name__ == "__main__":
    main()
