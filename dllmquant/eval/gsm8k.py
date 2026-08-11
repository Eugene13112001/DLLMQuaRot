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
# Ordered by how explicitly the model is pointing at its answer. A completion
# that ends "...he will eat 7 dozens of eggs in 4 weeks" has 4 as its last
# number, so the plain last-number rule scores a correct answer wrong -- and
# it does so silently, which is worse than a crash.
_MARKERS = [
    re.compile(r"\\boxed\{([^}]*)\}"),
    re.compile(r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\**\s*"
               r"(-?\$?\d[\d,]*\.?\d*)", re.IGNORECASE),
    re.compile(r"\*\*\s*(-?\$?\d[\d,]*\.?\d*)\s*\**"),
]


def _to_float(raw: str) -> Optional[float]:
    raw = raw.replace(",", "").replace("$", "").replace("\\", "").strip()
    raw = raw.rstrip(".").strip()
    m = _NUMBER.search(raw)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "").replace("$", "").rstrip("."))
    except ValueError:
        return None


def extract_answer(text: str) -> Optional[float]:
    """Pull the model's final answer out of a chain-of-thought completion.

    Prefers an explicit marker over position: \\boxed{}, then "the answer is
    N", then a bolded number, and only then the last number in the text.
    """
    if "####" in text:
        text = text.split("####")[-1]

    for pattern in _MARKERS:
        hits = pattern.findall(text)
        if hits:
            value = _to_float(hits[-1])
            if value is not None:
                return value

    matches = _NUMBER.findall(text)
    if not matches:
        return None
    return _to_float(matches[-1])


def gold_answer(answer_field: str) -> Optional[float]:
    return extract_answer(answer_field.split("####")[-1])


# A reply that ends on an operator, an open bracket or a dangling lead-in did
# not finish -- it ran out of canvas. Deliberately conservative: "The answer is
# 8" where the model meant 800 and lost the zeros looks complete and is not
# counted, so this is a floor on truncation, never an overestimate.
_CUT_OFF = re.compile(
    r"(?:[+\-*/=,:(]|\\times|\btimes\b|\bis\b|\bthe\b|\bof\b|\band\b|\bto\b)[\s$*]*$",
    re.IGNORECASE,
)


@dataclass
class EvalResult:
    correct: int
    total: int
    samples: List[dict]

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)

    @property
    def cut_off(self) -> int:
        """Replies that stop mid-expression, having run out of generation budget.

        A diffusion LM fills a canvas whose length was fixed before decoding
        started, so a reply that needs one more line does not get one -- it is
        scored wrong. That is a property of the budget, not of the model, and
        an accuracy reported without it invites reading a shortfall in
        `gen_length` as a shortfall in reasoning.
        """
        return sum(1 for s in self.samples if _CUT_OFF.search(s["completion"].rstrip()))

    @property
    def cut_off_wrong(self) -> int:
        return sum(
            1 for s in self.samples
            if not s["correct"] and _CUT_OFF.search(s["completion"].rstrip())
        )

    def summary(self) -> str:
        lines = [f"GSM8K: {self.correct}/{self.total} = {100 * self.accuracy:.2f}%"]
        wrong = self.total - self.correct
        if self.cut_off:
            share = 100 * self.cut_off_wrong / max(wrong, 1)
            lines.append(
                f"  {self.cut_off} replies were cut off mid-answer; they account "
                f"for {share:.0f}% of the errors -- raise --gen-length before "
                "reading this as accuracy"
            )
        return "\n".join(lines)


# Asking for an explicit marker is not cosmetic: without one, extraction has
# to guess which number in the closing sentence is the answer, and "7 dozens
# of eggs in 4 weeks" has no syntactic tell. Requiring the marker moves the
# problem from a heuristic to a contract.
ANSWER_INSTRUCTION = (
    "Think step by step, then end your reply with the final answer on its own "
    'line in exactly this form: "The answer is N", where N is a single number '
    "with no units."
)


def build_prompt(adapter: ModelAdapter, question: str, few_shot: str = "") -> torch.Tensor:
    tok = adapter.tokenizer
    text = f"{few_shot}Question: {question}\n{ANSWER_INSTRUCTION}\nAnswer:"
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(
            [{"role": "user", "content": f"{question}\n\n{ANSWER_INSTRUCTION}"}],
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
