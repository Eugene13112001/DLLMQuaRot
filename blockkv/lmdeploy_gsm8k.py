"""SDAR on GSM8K or RULER (--bench ruler) inside LMDeploy's PyTorch engine, with its real (packed) KV-cache quantization.

    python -m blockkv.lmdeploy_gsm8k --model-path $MODELS_DIR/SDAR-8B-Chat-b16 --block 16 \
        --threshold 0.9 --quant-policy 8 --out out/lmdeploy/sdar8b_b16_t0.9_q8.jsonl

--quant-policy: 0 none, 4 int4, 8 int8, 16 fp8 e4m3, 17 fp8 e5m2, 42 TurboQuant (K 4 bit, V 2 bit).

Same prompt, chat template, stop tokens, generation length (4096) and float16 as blockkv.run_gsm8k
for SDAR, and the same commit rule: LMDeploy's `low_confidence_dynamic` unmasks every masked token
with confidence >= threshold, or the single most confident one if none is.

Steps are counted by wrapping LMDeploy's UnmaskingProcessor: a step is a call in which at least one
token went from masked to unmasked (our `steps`); `calls` counts every call, including the pass that
moves a finished block into the cache. Per-problem counts need --batch 1 (one request at a time).
With --batch B > 1 the requests are served together: per-problem steps are not available, the
meta file gets the total wall time and the total number of calls.

Each line: idx, correct, pred, gold, gen_tokens, steps, calls, time_s, finish, text — readable by
blockkv.summarize (compare / table). An existing output is resumed (batch 1 only).
"""
import argparse
import json
import threading
import time
from pathlib import Path

from blockkv import gsm8k

MASKED = None   # set from lmdeploy.pytorch.consts once lmdeploy is imported


class StepCounter:
    """Counts unmasking calls and the calls that committed at least one token."""

    def __init__(self):
        self.lock = threading.Lock()
        self.calls = 0
        self.steps = 0
        self.committed = 0

    def snapshot(self):
        with self.lock:
            return self.calls, self.steps, self.committed

    def record(self, before, after, masked, unmasked):
        n = int(((before == masked) & (after == unmasked)).sum())
        with self.lock:
            self.calls += 1
            self.committed += n
            self.steps += n > 0


COUNTER = StepCounter()


def patch_unmasking():
    from lmdeploy.pytorch import consts
    from lmdeploy.pytorch.strategies.dllm import unmasking

    cls = unmasking.UnmaskingProcessor
    if getattr(cls, "_blockkv_patched", False):
        return
    original = cls.__call__

    def counted(self, logits, input_ids, token_ids, dllm_mask):
        before = dllm_mask.clone()          # the original edits the mask in place
        out = original(self, logits, input_ids, token_ids, dllm_mask)
        after = out[0] if isinstance(out, tuple) else out
        COUNTER.record(before, after, consts.DLLM_MASKED, consts.DLLM_UNMASKED)
        return out

    cls.__call__ = counted
    cls._blockkv_patched = True


def stop_ids(tok, model_path):
    """As blockkv.run_gsm8k for SDAR: tokenizer eos + generation_config eos."""
    from transformers import GenerationConfig as HFGen
    ids = set()
    try:
        e2 = HFGen.from_pretrained(model_path).eos_token_id
    except OSError:
        e2 = None
    for e in (tok.eos_token_id, e2):
        ids.update(e if isinstance(e, (list, tuple)) else [e] if e is not None else [])
    return sorted(ids)


def read_done(out):
    if not out.exists():
        return set()
    done, good = set(), []
    for line in out.read_text(encoding="utf-8").splitlines():
        try:
            done.add(json.loads(line)["idx"])
            good.append(line)
        except (json.JSONDecodeError, KeyError):
            break
    out.write_text("".join(l + "\n" for l in good), encoding="utf-8")
    return done


def items_gsm8k(tok, args):
    problems = gsm8k.load(args.data)[:args.limit]
    out = []
    for i, p in enumerate(problems):
        text = tok.apply_chat_template(
            [{"role": "user", "content": gsm8k.PROMPT_STYLES["sdar"].format(question=p["question"])}],
            tokenize=False, add_generation_prompt=True)

        def grade(answer, p=p):
            pred, ref = gsm8k.extract(answer), gsm8k.gold(p["answer"])
            return dict(correct=gsm8k.is_correct(pred, ref), pred=pred, gold=ref)
        out.append((i, text, args.gen_length, grade))
    return out


def items_ruler(tok, args):
    """The same RULER problems, prompt and scoring as blockkv.run_ruler (30 per task, seed 0)."""
    from blockkv.run_ruler import load, pick, prompt_text, score
    rows = load(args.ruler_data, args.length)
    out = []
    for i in pick(rows, args.per_task, args.seed)[:args.limit]:
        r = rows[i]

        def grade(answer, r=r):
            sc = score(r["task"], answer, r["answer"])
            return dict(task=r["task"], score=sc, correct=sc == 1.0)
        out.append((i, prompt_text(tok, r), int(r["max_new_tokens"]), grade))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--quant-policy", type=int, default=0, choices=[0, 4, 8, 16, 17, 42])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", choices=["gsm8k", "ruler"], default="gsm8k")
    ap.add_argument("--data", help="GSM8K test .jsonl or .parquet (default: HF openai/gsm8k)")
    ap.add_argument("--ruler-data", help="folder with the simonjegou/ruler parquet files (as for run_ruler)")
    ap.add_argument("--length", type=int, default=4096, help="RULER input length")
    ap.add_argument("--per-task", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--gen-length", type=int, default=4096, help="GSM8K answer limit (RULER: the task's own)")
    ap.add_argument("--batch", type=int, default=1, help="requests served together (1: per-problem steps)")
    ap.add_argument("--session-len", type=int, help="default: 8192 GSM8K, length + 4096 RULER")
    ap.add_argument("--cache-frac", type=float, default=0.3,
                    help="share of free GPU memory for the KV cache (the cards are shared)")
    ap.add_argument("--eager", action="store_true", help="no CUDA graphs")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "auto"],
                    help="float16 as in our runs (the SDAR authors' evaluation)")
    args = ap.parse_args(argv)

    from lmdeploy import GenerationConfig, PytorchEngineConfig, pipeline
    from transformers import AutoTokenizer
    patch_unmasking()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    stops = stop_ids(tok, args.model_path)
    items = items_gsm8k(tok, args) if args.bench == "gsm8k" else items_ruler(tok, args)
    session_len = args.session_len or (8192 if args.bench == "gsm8k" else args.length + 4096)

    cfg = PytorchEngineConfig(dtype=args.dtype, quant_policy=args.quant_policy, max_batch_size=args.batch,
                              session_len=session_len, cache_max_entry_count=args.cache_frac, eager_mode=args.eager,
                              dllm_block_length=args.block, dllm_unmasking_strategy="low_confidence_dynamic",
                              dllm_confidence_threshold=args.threshold)

    def gen(n):
        return GenerationConfig(max_new_tokens=n, do_sample=False, top_k=1, stop_token_ids=stops,
                                skip_special_tokens=True)

    import lmdeploy
    meta = dict(vars(args), lmdeploy=lmdeploy.__version__, stop_ids=stops, prompt_style="sdar",
                strategy="low_confidence_dynamic", session_len_used=session_len, n_items=len(items))
    t_start = time.time()
    with pipeline(args.model_path, backend_config=cfg) as pipe:
        if args.batch == 1:
            done = read_done(out)
            with open(out, "a", encoding="utf-8") as f:
                for i, text, n_new, grade in items:
                    if i in done:
                        continue
                    c0, s0, n0 = COUNTER.snapshot()
                    t0 = time.time()
                    r = pipe(text, gen_config=gen(n_new), do_preprocess=False)
                    dt = time.time() - t0
                    c1, s1, n1 = COUNTER.snapshot()
                    if c1 == c0:
                        raise SystemExit("step counter did not move: the engine runs the unmasking outside "
                                         "this process, so the patch is not seen. Report this.")
                    row = dict(idx=i, **grade(r.text), gen_tokens=r.generate_token_len,
                               prompt_tokens=r.input_token_len, steps=s1 - s0, calls=c1 - c0,
                               committed=n1 - n0, time_s=round(dt, 4), finish=r.finish_reason, text=r.text)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
        else:
            c0, s0, n0 = COUNTER.snapshot()
            t0 = time.time()
            rs = pipe([t for _, t, _, _ in items], gen_config=[gen(n) for _, _, n, _ in items], do_preprocess=False)
            dt = time.time() - t0
            c1, s1, n1 = COUNTER.snapshot()
            with open(out, "w", encoding="utf-8") as f:
                for (i, _, _, grade), r in zip(items, rs):
                    f.write(json.dumps(dict(idx=i, **grade(r.text), gen_tokens=r.generate_token_len,
                                            prompt_tokens=r.input_token_len, steps=0, finish=r.finish_reason,
                                            text=r.text), ensure_ascii=False) + "\n")
            tokens = sum(r.generate_token_len for r in rs)
            meta.update(total_time_s=round(dt, 3), total_calls=c1 - c0, total_steps=s1 - s0,
                        total_gen_tokens=tokens, tokens_per_s=round(tokens / dt, 2),
                        requests_per_s=round(len(rs) / dt, 4))
    meta["wall_s"] = round(time.time() - t_start, 1)
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
