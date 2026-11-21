"""Evaluate the supplied QuMix quantizer using the project's SDAR decoder."""

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch

from blockkv import decode, longbench, run_longproc
from blockkv.kvquant import KVQuant
from blockkv.run_gsm8k import device_info, mask_id, read_done, stop_ids
from third_party.qumix.qumix.reference import entry_to_bits


AUTHOR = Path(__file__).resolve().parents[1] / "third_party/qumix/accuracy/sdar/qumix_eval.py"
ARMS = ("base", "kivis2r128", "qmxmatchkivi2ts")
LONGBENCH_ARMS = ("base", "kivi_local2", "qmxmatchkivi2ts")
TASKS = ("qmsum", "gov_report")
PROCEDURES = ("path_traversal_2k", "html_to_tsv_2k", "tom_tracking_2k",
              "countdown_2k", "travel_planning_2k")


def load_author(model, calibration, block):
    # The supplied script initializes the model and installs its cache hooks
    # at module scope. Stop before its dataset and evaluation loop.
    source = AUTHOR.read_text(encoding="utf-8")
    setup, marker, _ = source.partition("# ========== [evaluation] datasets and metrics")
    if not marker:
        raise RuntimeError("Could not locate the dataset boundary in the QuMix evaluator.")
    namespace = {"__file__": str(AUTHOR), "__name__": "_qumix_author"}
    argv = sys.argv
    try:
        sys.argv = [str(AUTHOR), "--model", model, "--loadcalib", str(calibration),
                    "--block", str(block), "--steps", str(block), "--prerope", "--vbits", "2",
                    "--vsbits", "8", "--arms", ",".join(ARMS)]
        exec(compile(setup, str(AUTHOR), "exec"), namespace)
    finally:
        sys.argv = argv
    return namespace


def limit_store_logits(model):
    # The decoder discards logits during prefill and completed-block writes.
    # Keep the full logits on denoising forwards for the confidence threshold.
    if "logits_to_keep" not in inspect.signature(model.forward).parameters:
        return

    def hook(_model, inputs, kwargs):
        if kwargs.get("store_kv"):
            kwargs.setdefault("logits_to_keep", 1)
        return inputs, kwargs

    model.register_forward_pre_hook(hook, with_kwargs=True)


def evaluate(args):
    arms = LONGBENCH_ARMS if args.bench == "longbench" else ARMS
    settings = dict(bench=args.bench, data=str(args.data.resolve()), model=args.model,
                    block=args.block, threshold=0.9, dtype="bfloat16", limit=args.limit,
                    calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest())
    outputs = {}
    args.out.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        path = args.out / f"sdar8b_b{args.block}_{arm}.jsonl"
        meta = path.with_suffix(".meta.json")
        fingerprint = dict(settings, arm=arm)
        if path.is_file() and path.stat().st_size:
            if not meta.is_file() or json.loads(meta.read_text(encoding="utf-8"))["run"] != fingerprint:
                raise SystemExit(f"Existing output has different settings: {path}")
        outputs[arm] = path, meta, fingerprint

    if args.bench == "longbench":
        longbench.require_metrics()
        work = longbench.items(TASKS, args.data, None, 16384)[:args.limit]
    else:
        work = run_longproc.load(args.data, PROCEDURES, 20, 0)[:args.limit]

    author = load_author(args.model, args.calibration, args.block)
    model, tok = author["model"], author["tok"]
    limit_store_logits(model)
    mid, stops = mask_id("sdar", tok, None), stop_ids("sdar", tok, model)
    layers, head_dim = author["L"], author["HD"]
    tag = ARMS[-1].removesuffix("ts")
    proj = author["PROJ"]
    if set(proj) != set(range(layers)) or any(tag not in layer for layer in proj.values()):
        raise SystemExit(f"Incomplete QuMix allocation for {tag} in {args.calibration}")
    qmx_key_rates = []
    for layer in range(layers):
        _, _, bits = entry_to_bits(proj[layer][tag])
        qmx_key_rates.append((bits + 0.5 * ((bits > 0) & (bits < 16))).mean().item())
    for arm, (path, meta, fingerprint) in outputs.items():
        author["ARM"][0] = "base" if arm == "kivi_local2" else arm
        done = read_done(path)
        meta.write_text(json.dumps(dict(run=fingerprint, device=device_info()), indent=2),
                        encoding="utf-8")
        with path.open("a", encoding="utf-8") as stream:
            for idx, task, *example in work:
                if idx in done:
                    continue
                if args.bench == "longbench":
                    (row,) = example
                    prompt = longbench.chat_prompt(task, row, tok, 16384)
                    limit = longbench.MAX_NEW[task]
                else:
                    text, limit, scorer, row = example
                    prompt = tok.apply_chat_template([{"role": "user", "content": text}],
                                                     tokenize=False, add_generation_prompt=True)
                ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
                if args.bench == "longproc" and ids.shape[1] > 24576:
                    print(f"{arm} {idx}: skipped, prompt={ids.shape[1]}", flush=True)
                    continue
                local_quant = KVQuant(2) if arm == "kivi_local2" else None
                gen, stats = decode.sdar_generate(
                    model, ids.cuda(), mid, args.block, args.block,
                    0.9, limit, stops, on_store=local_quant,
                )
                gen = gen[:limit]
                answer = tok.decode(gen, skip_special_tokens=True)
                if args.bench == "longbench":
                    metrics = longbench.scores(answer, [str(a) for a in row["answers"]])
                    score = metrics["score"]
                else:
                    try:
                        metrics, _ = scorer(answer, row)
                    except Exception as error:
                        metrics = {"accuracy": 0.0, "scorer_error": str(error)[:200]}
                    score = run_longproc.main_score({k: v for k, v in metrics.items()
                                                     if isinstance(v, (int, float))})
                cache_len = stats.cache_len
                saved = 0.0
                if cache_len and arm in ARMS[1:]:
                    for layer in range(layers):
                        keys = author["SEALED"].get(layer, 0)
                        if arm == "kivis2r128":
                            values = author["SEALEDV"].get(layer, 0)
                            saved += (16 - 3) * keys + (16 - 2 - 32 / head_dim) * values
                        else:
                            saved += (32 - qmx_key_rates[layer] - 2 - 16 / head_dim) * keys
                bits_avg = 16 - saved / (2 * layers * max(1, cache_len))
                result = dict(idx=idx, task=task, score=score, correct=score, metrics=metrics,
                              prompt_tokens=ids.shape[1], gen_tokens=len(gen), steps=stats.steps,
                              per_step=stats.per_step, kv_tokens=cache_len,
                              kv_bits_avg=round(bits_avg, 4), text=answer)
                if local_quant is not None:
                    result.update(local_quant.summary(cache_len))
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{arm} {idx}: steps={stats.steps} tokens={len(gen)} score={score:.3f}",
                      flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True, choices=("longbench", "longproc"))
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True, help="SDAR-8B checkpoint for the selected block")
    parser.add_argument("--block", type=int, choices=(4, 8, 16, 32), default=32)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="limit total examples for a smoke run")
    args = parser.parse_args()
    if not args.data.exists() or not args.calibration.is_file():
        parser.error("--data and --calibration must exist")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    evaluate(args)


if __name__ == "__main__":
    main()
