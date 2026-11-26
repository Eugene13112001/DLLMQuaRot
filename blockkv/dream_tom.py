"""Unquantized DreamReasoner baseline on the official LongProc ToM 8K split."""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

from blockkv.tax_decompose import repetitive


MODEL = "Dream-org/DreamReasoner-8B"
REV = "ed62b1d2c82ccd234b05ed2463b4c0ee640f2068"
TASK = "tom_tracking_8k"
CAP = 9216


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--longproc", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.longproc.resolve()))
    from longproc.longproc_data import load_longproc_data

    data, score = load_longproc_data(TASK, str(args.longproc / "data"))
    if not data:
        raise ValueError("Empty LongProc split")
    args.out.mkdir(parents=True, exist_ok=True)
    settings = dict(revision=REV, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL, **settings)
    model = None
    rows = []
    for i, item in enumerate(data):
        path = args.out / f"item_{i:04d}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL, torch_dtype=torch.bfloat16, **settings).to("cuda").eval()
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": item["input_prompt"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        if ids.shape[1] + CAP > model.config.max_position_embeddings:
            raise ValueError(f"Context window exceeded on {TASK}:{i}")
        with torch.inference_mode():
            result = model.block_diffusion_generate(
                input_ids=ids, max_new_tokens=CAP, block_length=32, temperature=0.0,
                confidence_threshold=0.9, use_kv_cache=True, return_dict_in_generate=True)
        generated = result.sequences[0, ids.shape[1]:].tolist()
        stops = {model.config.eos_token_id, tok.eos_token_id}
        end = next((k for k, token in enumerate(generated) if token in stops), len(generated))
        text = tok.decode(generated[:end], skip_special_tokens=True)
        metrics, details = score(text, item)
        row = dict(idx=f"{TASK}:{i}", model=MODEL, revision=REV, block=32, threshold=0.9,
                   thinking=False, dtype="bfloat16", transformers=transformers.__version__,
                   max_new_tokens=CAP, prompt_tokens=ids.shape[1], gen_tokens=end,
                   reference_tokens=len(tok.encode(item["reference_output"], add_special_tokens=False)),
                   stop_token=generated[end] if end < len(generated) else None,
                   at_cap=end == CAP, repetitive_tail=repetitive(text), nfe=int(result.nfe),
                   metrics=metrics, details=details, text=text)
        save(path, row)
        rows.append(row)
        print(f"[{i+1}/{len(data)}] {row['idx']} tokens={end} metrics={metrics}", flush=True)
    summary = dict(n=len(rows), metrics={k: mean(r["metrics"][k] for r in rows)
                                         for k in ("accuracy", "partial_accuracy")},
                   median_tokens=median(r["gen_tokens"] for r in rows),
                   median_reference_tokens=median(r["reference_tokens"] for r in rows),
                   reached_8192=sum(r["gen_tokens"] >= 8192 for r in rows),
                   at_cap=sum(r["at_cap"] for r in rows),
                   repetitive_tail=sum(r["repetitive_tail"] for r in rows))
    save(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
