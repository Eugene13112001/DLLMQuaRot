"""LongBench summarization tasks (long input, answers of hundreds of tokens): qmsum, gov_report.

Data: LongBench's `data.zip` (HF dataset zai-org/LongBench, formerly THUDM/LongBench) or a folder
with its `<task>.jsonl` files. Prompts and answer limits are LongBench's own (config/dataset2prompt.json,
dataset2maxlen.json). A prompt longer than `max_tokens` is cut in the middle, keeping the first and
the last half, as LongBench's pred.py does. Score: best ROUGE-L F over the references, with Google's
rouge_score and stemming, as BitSieve scores qmsum.
"""
import json
import zipfile
from pathlib import Path

PROMPTS = {
    "qmsum": ("You are given a meeting transcript and a query containing a question or instruction. "
              "Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query "
              "based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:"),
    "gov_report": ("You are given a report by a government agency. Write a one-page summary of the report."
                   "\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"),
}
MAX_NEW = {"qmsum": 512, "gov_report": 512}


def load(task, path):
    """Rows of one task, from data.zip or a folder of jsonl files."""
    path = Path(path).expanduser()
    if path.is_file():
        with zipfile.ZipFile(path) as zf, zf.open(f"data/{task}.jsonl") as f:
            lines = f.read().decode("utf-8").splitlines()
    else:
        files = list(path.rglob(f"{task}.jsonl"))
        if not files:
            raise SystemExit(f"no {task}.jsonl under {path}")
        lines = files[0].read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def raw_prompt(task, row, tok, max_tokens):
    """LongBench prompt, cut in the middle to at most `max_tokens` tokens (no chat template yet)."""
    text = PROMPTS[task].format(context=row["context"], input=row.get("input", ""))
    ids = tok(text, add_special_tokens=False).input_ids
    if max_tokens and len(ids) > max_tokens:
        half = max_tokens // 2
        text = tok.decode(ids[:half], skip_special_tokens=True) + tok.decode(ids[-half:], skip_special_tokens=True)
    return text


def chat_prompt(task, row, tok, max_tokens):
    return tok.apply_chat_template([{"role": "user", "content": raw_prompt(task, row, tok, max_tokens)}],
                                   tokenize=False, add_generation_prompt=True)


def rouge_l(pred, ref):
    try:
        from rouge_score import rouge_scorer
    except ImportError:              # plain LCS F-measure on whitespace tokens (tests only)
        p, r = pred.split(), ref.split()
        if not p or not r:
            return 0.0
        row = [0] * (len(r) + 1)
        for a in p:
            prev = 0
            for j, b in enumerate(r, 1):
                old = row[j]
                row[j] = prev + 1 if a == b else max(row[j], row[j - 1])
                prev = old
        return 2 * row[-1] / (len(p) + len(r))
    return float(rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(ref, pred)["rougeL"].fmeasure)


def score(pred, refs):
    return max((rouge_l(pred, r) for r in refs), default=0.0)


def items(tasks, path, tok, max_tokens, per_task=None):
    """[(idx, task, row)]; idx = "<task>:<line>" so both tasks share one output file."""
    out = []
    for t in tasks:
        rows = load(t, path)[:per_task]
        out += [(f"{t}:{i}", t, r) for i, r in enumerate(rows)]
    return out
