"""Paired comparison of two BitSieve quality runs (their own code, their GSM8K prompt and scoring).

    python -m blockkv.bitsieve_quality REF.jsonl RUN.jsonl

REF/RUN are the per-problem files written by `bitsieve_fastdllm.eval.quality`
(e.g. official_dense_bf16.jsonl and dense_kivi2_k2v2_r32.jsonl). Numbers are the same as
`blockkv.summarize.compare`: accuracy with a paired ±95%, steps per problem, tokens per step.
An answer that never produced a stop token counts as a runaway.
"""
import json
import sys

from blockkv.summarize import RUNAWAY, paired


def convert(row):
    rt = row["runtime"]
    gen = int(rt["generated_tokens_per_request"])
    return dict(correct=float(row["score"]) > 0.5, steps=int(sum(rt["denoising_steps_per_block"])),
                gen_tokens=gen if rt.get("stop_found", True) else max(gen, RUNAWAY),
                text=row["prediction"])


def read(path):
    with open(path, encoding="utf-8") as f:
        return {str(r["id"]): convert(r) for r in map(json.loads, filter(str.strip, f))}


def main(argv=None):
    a, b = (argv or sys.argv[1:])[:2]
    p = paired(read(a), read(b))
    print(json.dumps(dict(ref=a, run=b, n=p["n"], acc_ref=round(100 * p["acc_a"], 2), acc_run=round(100 * p["acc_b"], 2),
                          diff=round(100 * (p["acc_b"] - p["acc_a"]), 2), ci95=round(196 * p["se"], 2),
                          only_ref=p["only_a"], only_run=p["only_b"], runaway_ref=p["runaway_a"], runaway_run=p["runaway_b"],
                          steps_ref=round(p["steps_a"], 2), steps_run=round(p["steps_b"], 2),
                          tax_pct=round(100 * (p["steps_b"] / p["steps_a"] - 1), 2),
                          med_steps_ref=p["med_steps_a"], med_steps_run=p["med_steps_b"],
                          tps_ref=round(p["tps_a"], 3), tps_run=round(p["tps_b"], 3)), ensure_ascii=False))


if __name__ == "__main__":
    main()
