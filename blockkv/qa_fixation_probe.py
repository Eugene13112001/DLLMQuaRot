"""Independent, paired leave-one-out probes on an unchanged BF16 SDAR QA trajectory."""

import argparse
import copy
import json
import math
from pathlib import Path

import torch

from blockkv import decode, longbench
from blockkv.kvquant import KVQuant
from blockkv.run_gsm8k import mask_id, repo, resolve_path, stop_ids


class BlockObserver:
    def __init__(self, model, mid, prompt_length, stops):
        self.model = model
        self.mid = mid
        self.prompt_length = prompt_length
        self.stops = stops
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
        allowed &= kw["position_ids"][0] < self.prompt_length + 512
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
    parser.add_argument("--block", type=int, required=True, choices=(4, 8, 16, 32))
    parser.add_argument("--per-task", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.per_task <= 200:
        parser.error("--per-task must be between 1 and 200")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    family, name = repo("sdar8b", args.block)
    path = resolve_path(name, None)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True, device_map="cuda").eval()
    mid, stops = mask_id(family, tok, None), stop_ids(family, tok, model)
    root = Path("out/diagnostics/qa_fixation_v1") / f"n{args.per_task}"
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"sdar8b_b{args.block}.jsonl"
    import fcntl
    with out.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        done = {json.loads(line)["idx"] for line in stream}
        for idx, task, data in longbench.items(longbench.QA_TASKS, args.data, None, 16384, args.per_task):
            if idx in done:
                continue
            prompt = longbench.chat_prompt(task, data, tok, 16384)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            observer = BlockObserver(model, mid, ids.shape[1], stops)
            gen, stats = decode.sdar_generate(observer, ids, mid, args.block, args.block,
                                               0.9, 512, stops)
            record = dict(idx=idx, task=task, block=args.block, checkpoint=str(path),
                          revision=getattr(model.config, "_commit_hash", None),
                          prompt_tokens=ids.shape[1], generated=gen[:512].tolist(),
                          steps=stats.steps, blocks=observer.blocks, probes=observer.rows)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(f"{idx} steps={stats.steps} paired_positions={len(observer.rows)}", flush=True)


if __name__ == "__main__":
    main()
