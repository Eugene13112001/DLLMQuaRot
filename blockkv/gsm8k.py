"""GSM8K loading, prompting and scoring."""
import json
import math
import re
from pathlib import Path

# User message per model, as in the authors' own GSM8K evaluation.
#   sdar:     OpenCompass gsm8k_0shot_v2_gen_17d799 (SDAR evaluation/opencompass configs).
#   fastdllm: lm-eval `gsm8k` 0-shot, "Question: ...\nAnswer:", with "Answer:" replaced
#             in Fast-dLLM v2/eval.py by a plain (non-format) string, so the braces stay doubled.
PROMPT_STYLES = {
    "sdar": "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    "fastdllm": "Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{{{}}}}.",
    "plain": "{question}",
}

_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def load(path=None):
    """Return the 1319 test problems as dicts with 'question' and 'answer'.

    `path` may be the grade-school-math test.jsonl or the HF parquet file;
    without it the `openai/gsm8k` dataset is loaded through `datasets`.
    """
    if path is None:
        from datasets import load_dataset
        return [dict(r) for r in load_dataset("openai/gsm8k", "main", split="test")]
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gold(answer):
    """The reference number after '####'."""
    return normalize(answer.split("####")[-1])


def normalize(s):
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        x = float(s)
    except ValueError:
        return s
    if not math.isfinite(x):      # a runaway answer: more digits than a float holds, or 'nan'
        return s
    return str(int(x)) if x == int(x) else str(x)


def _last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k]
    return None


def extract(text):
    """The model's final number: last \\boxed{}, then '####', 'answer is', then the last number."""
    boxed = _last_boxed(text)
    if boxed is not None:
        nums = re.findall(_NUM, boxed)
        return normalize(nums[-1]) if nums else normalize(boxed)
    for pat in (r"####\s*(" + _NUM + ")", r"answer is[:\s$]*(" + _NUM + ")"):
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m:
            return normalize(m[-1])
    nums = re.findall(_NUM, text)
    return normalize(nums[-1]) if nums else ""


def is_correct(pred, ref):
    if pred == ref:
        return True
    try:
        return abs(float(pred) - float(ref)) < 1e-6
    except ValueError:
        return False
