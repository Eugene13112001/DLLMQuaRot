"""GSM8K accuracy for a (quantized) DLLM.

GSM8K is the benchmark where the paper reports its largest gap -- 10+ points on
LLaDA at 4 bits -- because multi-step arithmetic is exactly what a badly
calibrated quantizer destroys first: one wrong committed token early in the
trajectory cannot be revised later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from ..config import TMASConfig
from ..models.base import ModelAdapter

_NUMBER = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> Optional[float]:
    """Take the last number in the completion, GSM8K convention."""
    if "####" in text:
        text = text.split("####")[-1]
    matches = _NUMBER.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(raw)
    except ValueError:
        return None


def gold_answer(answer_field: str) -> Optional[float]:
    return extract_answer(answer_field.split("####")[-1])


@dataclass
class EvalResult:
    correct: int
    total: int
    samples: List[dict]

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)

    def summary(self) -> str:
        return f"GSM8K: {self.correct}/{self.total} = {100 * self.accuracy:.2f}%"


def build_prompt(adapter: ModelAdapter, question: str, few_shot: str = "") -> torch.Tensor:
    text = f"{few_shot}Question: {question}\nAnswer:"
    tok = adapter.tokenizer
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(
            [{"role": "user", "content":
              f"{question}\n\nThink step by step, then give the final number."}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return tok(text, return_tensors="pt")["input_ids"][0]


@torch.no_grad()
def evaluate_gsm8k(
    adapter: ModelAdapter,
    n_samples: int = 200,
    gen_cfg: Optional[TMASConfig] = None,
    split: str = "test",
    verbose: bool = True,
) -> EvalResult:
    from datasets import load_dataset

    cfg = gen_cfg or TMASConfig(gen_length=256, block_length=32, steps=256)
    ds = load_dataset("gsm8k", "main", split=split)
    n = min(n_samples, len(ds))

    correct = 0
    samples: List[dict] = []

    for i in range(n):
        item = ds[i]
        prompt = build_prompt(adapter, item["question"])
        out = adapter.generate(prompt, cfg)
        completion = adapter.tokenizer.decode(
            out[0, prompt.shape[-1]:], skip_special_tokens=True
        )

        pred = extract_answer(completion)
        gold = gold_answer(item["answer"])
        hit = pred is not None and gold is not None and abs(pred - gold) < 1e-4
        correct += int(hit)

        samples.append(
            {"question": item["question"], "completion": completion,
             "pred": pred, "gold": gold, "correct": hit}
        )
        if verbose and (i + 1) % 20 == 0:
            print(f"  [gsm8k] {i + 1}/{n}  running acc {100 * correct / (i + 1):.1f}%")

    return EvalResult(correct=correct, total=n, samples=samples)


__all__ = ["evaluate_gsm8k", "EvalResult", "extract_answer", "gold_answer"]
