"""Independent, paired leave-one-out probes on an unchanged BF16 SDAR QA trajectory."""

import argparse
import copy
import json
import math
from pathlib import Path

import torch

from blockkv import decode, gsm8k, longbench
from blockkv.kvquant import KVQuant
from blockkv.run_gsm8k import mask_id, repo, resolve_path, stop_ids


class BlockObserver:
    def __init__(self, model, mid, prompt_length, stops, gen_limit=512):
        self.model = model
        self.mid = mid
        self.prompt_length = prompt_length
        self.stops = stops
        self.gen_limit = gen_limit
        self.states = []
        self.rows = []
        self.blocks = []

    def __call__(self, cur, **kw):
        if kw["store_kv"]:
            if self.states:
                self.inspect(cur, kw)
                self.states.clear()
        else:
            self.states.append(cur.clone())
        return self.model(cur, **kw)

    def inspect(self, final, kw):
        states = [*self.states, final]
        initial = states[0][0].eq(self.mid)
        slots = int(initial.sum())
        filled = [int((initial & s[0].ne(self.mid)).sum()) for s in states]
        start = int(kw["position_ids"][0, 0])
        allowed = initial.clone()
        eos = decode.first_stop(final[0, initial], self.stops)
        generated = initial.nonzero().flatten()
        if eos is not None:
            allowed[generated[eos:]] = False
        allowed &= kw["position_ids"][0] < self.prompt_length + self.gen_limit
        block = dict(start=start, slots=slots, eos_slot=eos, eligible_positions=0)
        self.blocks.append(block)
        quant_cache = None
        for early in range(1, len(states)):
            if 4 * filled[early] > slots:
                break
            late = next((j for j in range(early + 1, len(states))
                         if filled[j] - filled[early] >= math.ceil(slots / 2)), None)
            if late is None:
                continue
            fresh = states[early-1][0].eq(self.mid) & states[early][0].ne(self.mid) & allowed
            for pos in fresh.nonzero().flatten().tolist():
                if quant_cache is None:
                    quant_cache = copy.deepcopy(kw["past_key_values"])
                    KVQuant(2)(quant_cache, 0, start)
                token = int(final[0, pos])
                row = dict(start=start, pos=pos, token=token, slots=slots,
                           early_fill=filled[early], late_fill=filled[late],
                           early_step=early, late_step=late)
                for label, state in (("early", states[early]), ("late", states[late])):
                    hidden = state.clone()
                    hidden[0, pos] = self.mid
                    for arm, cache in (("bf16", kw["past_key_values"]), ("kv2", quant_cache)):
                        opts = dict(kw, past_key_values=cache, store_kv=False)
                        logits = self.model(hidden, **opts).logits[0, pos].float()
                        probs = logits.softmax(-1)
                        row[f"{label}_{arm}"] = dict(top=int(probs.argmax()), p=float(probs[token]))
                self.rows.append(row)
                block["eligible_positions"] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bench", choices=("longbench", "longbench-summary", "gsm8k"), default="longbench")
    parser.add_argument("--block", type=int, required=True, choices=(4, 8, 16, 32))
    parser.add_argument("--per-task", type=int, default=20)
    parser.add_argument("--prefill-chunk", type=int, help="prefill in block-aligned chunks of this many tokens")
    args = parser.parse_args()
    if args.prefill_chunk is not None and (args.prefill_chunk < args.block or args.prefill_chunk % args.block):
        parser.error("--prefill-chunk must be a positive multiple of --block")
    maximum = 1319 if args.bench == "gsm8k" else 200
    if not 1 <= args.per_task <= maximum:
        parser.error(f"--per-task must be between 1 and {maximum}")
    if args.bench == "gsm8k":
        rows = gsm8k.load(args.data)
        if len(rows) < args.per_task:
            parser.error("GSM8K file has fewer examples than requested")
        work = [(i, "gsm8k", row) for i, row in enumerate(rows[:args.per_task])]
        gen_limit = 4096
        directory = "gsm8k_fixation_v1_g4096"
    else:
        summary = args.bench == "longbench-summary"
        tasks = ("qmsum", "gov_report") if summary else longbench.QA_TASKS
        work = longbench.items(tasks, args.data, None, 16384, args.per_task)
        gen_limit = 512
        directory = "summary_fixation_v1" if summary else "qa_fixation_v1"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    family, name = repo("sdar8b", args.block)
    path = resolve_path(name, None)
    root = Path("out/diagnostics") / directory / f"n{args.per_task}"
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"sdar8b_b{args.block}.jsonl"
    import fcntl
    with out.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        done = {json.loads(line)["idx"] for line in stream}
        if all(idx in done for idx, _, _ in work):
            print(f"Already complete: {out}", flush=True)
            return
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True, device_map="cuda").eval()
        mid, stops = mask_id(family, tok, None), stop_ids(family, tok, model)
        for idx, task, data in work:
            if idx in done:
                continue
            if task == "gsm8k":
                text = gsm8k.PROMPT_STYLES["sdar"].format(question=data["question"])
                prompt = tok.apply_chat_template([{"role": "user", "content": text}],
                                                 tokenize=False, add_generation_prompt=True)
            else:
                prompt = longbench.chat_prompt(task, data, tok, 16384)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            observer = BlockObserver(model, mid, ids.shape[1], stops, gen_limit)
            gen, stats = decode.sdar_generate(observer, ids, mid, args.block, args.block,
                                               0.9, gen_limit, stops, prefill_chunk=args.prefill_chunk)
            record = dict(idx=idx, task=task, block=args.block, checkpoint=str(path),
                          revision=getattr(model.config, "_commit_hash", None),
                          prompt_tokens=ids.shape[1], generated=gen[:gen_limit].tolist(),
                          steps=stats.steps, blocks=observer.blocks, probes=observer.rows)
            record.update(gen_limit=gen_limit, threshold=0.9, prefill_chunk=args.prefill_chunk,
                          peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(f"{task} id={idx} steps={stats.steps} paired_positions={len(observer.rows)} "
                  f"peak_allocated={record['peak_allocated_gib']:.2f} GiB "
                  f"peak_reserved={record['peak_reserved_gib']:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
