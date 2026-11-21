"""LongBench summarization and question answering.

Data: LongBench's `data.zip` (HF dataset zai-org/LongBench, formerly THUDM/LongBench) or a folder
with its `<task>.jsonl` files. Prompts and answer limits are LongBench's own (config/dataset2prompt.json,
dataset2maxlen.json). A prompt longer than `max_tokens` is cut in the middle, keeping the first and
the last half, as LongBench's pred.py does.

Scores are best over the references. Summarization reports two ROUGE-L scores:
  score       LongBench's own metric (metrics.rouge_score: the `rouge` package, no stemming), so the
              numbers compare with LongBench tables;
  score_bs    Google's rouge_score with stemming, as BitSieve scores qmsum.
Question answering reports LongBench token F1 as score.
Rescore saved answers: python -m blockkv.longbench rescore DATA FILE.jsonl...
"""
import json
import re
import string
import zipfile
from collections import Counter
from pathlib import Path

PROMPTS = {
    "qmsum": ("You are given a meeting transcript and a query containing a question or instruction. "
              "Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query "
              "based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:"),
    "gov_report": ("You are given a report by a government agency. Write a one-page summary of the report."
                   "\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"),
}
MAX_NEW = {"qmsum": 512, "gov_report": 512}
QA_TASKS = ("2wikimqa", "narrativeqa", "musique")
# Official templates from LongBench (see third_party/longbench/LICENSE).
_QA_PROMPT = (
    "Answer the question based on the given passages. Only give me the answer and do not output any other words."
    "\n\nThe following are given passages.\n{context}\n\n"
    "Answer the question based on the given passages. Only give me the answer and do not output any other words."
    "\n\nQuestion: {input}\nAnswer:"
)
PROMPTS.update({
    "2wikimqa": _QA_PROMPT,
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question asconcisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\nStory: {context}\n\n"
        "Now, answer the question based on the story asconcisely as you can, using a single phrase "
        "if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "musique": _QA_PROMPT,
})
MAX_NEW.update({"2wikimqa": 32, "narrativeqa": 128, "musique": 32})


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


def rouge_l_longbench(pred, ref):
    """LongBench metrics.rouge_score: Rouge().get_scores([pred], [ref], avg=True)["rouge-l"]["f"], 0 on error."""
    try:
        from rouge import Rouge
    except ImportError:          # tests without the package: whitespace LCS F-measure, unstemmed
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
    try:
        return float(Rouge().get_scores([pred], [ref], avg=True)["rouge-l"]["f"])
    except Exception:            # LongBench returns 0 for empty or unparsable text
        return 0.0


def score(pred, refs):
    """LongBench's official score."""
    return max((rouge_l_longbench(pred, r) for r in refs), default=0.0)


def score_bs(pred, refs):
    """BitSieve's score (rouge_score, stemming)."""
    return max((rouge_l(pred, r) for r in refs), default=0.0)


def qa_f1(pred, ref):
    def tokens(text):
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\b(a|an|the)\b", " ", text).split()

    prediction, reference = tokens(pred), tokens(ref)
    overlap = sum((Counter(prediction) & Counter(reference)).values())
    return 2 * overlap / (len(prediction) + len(reference)) if overlap else 0.0


def scores(pred, refs, task=None):
    if task in QA_TASKS:
        return dict(score=max((qa_f1(pred, ref) for ref in refs), default=0.0))
    return dict(score=score(pred, refs), score_bs=score_bs(pred, refs))


def rescore(data, files, *, validate_metrics=False):
    """Recompute the scores of saved answers (rows keep `text` and `idx` = "<task>:<line>")."""
    import sys
    cache = {}
    for f in files:
        f = Path(f)
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if validate_metrics and any(r["idx"].split(":", 1)[0] not in QA_TASKS for r in rows):
            require_metrics()
        for r in rows:
            task, line = r["idx"].split(":")
            if task not in cache:
                cache[task] = load(task, data)
            refs = [str(a) for a in cache[task][int(line)]["answers"]]
            r.update(scores(r["text"], refs, task))
            r["correct"] = r["score"]
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        tasks = dict.fromkeys(row["idx"].split(":", 1)[0] for row in rows)
        if not any(task in QA_TASKS for task in tasks):
            mean = lambda k: sum(r[k] for r in rows) / max(len(rows), 1)
            print(f"{f}: n={len(rows)} rougeL_longbench={mean('score'):.4f} "
                  f"rougeL_bitsieve={mean('score_bs'):.4f}", file=sys.stdout)
            continue
        for task in tasks:
            subset = [row for row in rows if row["idx"].startswith(task + ":")]
            mean = lambda k: sum(row[k] for row in subset) / len(subset)
            metric = (f"f1={mean('score'):.4f}" if task in QA_TASKS else
                      f"rougeL_longbench={mean('score'):.4f} rougeL_bitsieve={mean('score_bs'):.4f}")
            print(f"{f}: {task} n={len(subset)} {metric}", file=sys.stdout)


def require_metrics():
    """Real runs need both packages: the fallbacks above are for the tests only."""
    try:
        import rouge  # noqa: F401
        import rouge_score  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"{e}: pip install rouge rouge-score (LongBench's and BitSieve's ROUGE-L)")


def items(tasks, path, tok, max_tokens, per_task=None):
    """[(idx, task, row)]; idx = "<task>:<line>" so both tasks share one output file."""
    if tok is not None and any(task not in QA_TASKS for task in tasks):
        require_metrics()
    out = []
    for t in tasks:
        rows = load(t, path)[:per_task]
        out += [(f"{t}:{i}", t, r) for i, r in enumerate(rows)]
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4 or sys.argv[1] != "rescore":
        raise SystemExit("python -m blockkv.longbench rescore DATA FILE.jsonl...")
    rescore(sys.argv[2], sys.argv[3:], validate_metrics=True)
