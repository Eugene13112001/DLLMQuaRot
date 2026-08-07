"""Calibration prompts.

TMAS decides *which states* of the denoising trajectory to keep; this module
decides which prompts those trajectories start from.  Both matter -- a set of
prompts that never elicits reasoning produces a calibration set with no
chain-of-thought activations, which is where the paper's GSM8K gap shows up.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

# A small, dependency-free mix: instruction following, arithmetic reasoning,
# code, and free-form text.  Used when no HF dataset is available offline.
FALLBACK_PROMPTS: List[str] = [
    "Explain why the sky appears blue during the day and red at sunset.",
    "A train leaves at 14:20 and arrives at 17:05. How long is the journey?",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Summarise the causes of the 1929 stock market crash in three sentences.",
    "If a rectangle has perimeter 34 and width 6, what is its area?",
    "Translate to French: 'The library closes at eight on weekdays.'",
    "What is the difference between a list and a tuple in Python?",
    "Describe how a heat pump moves energy against a temperature gradient.",
    "Sarah has 3 boxes with 12 pencils each. She gives away 17. How many remain?",
    "Write a SQL query returning the second highest salary from a table.",
    "Explain the bias-variance tradeoff to someone who knows basic statistics.",
    "A recipe for 4 people needs 300g of flour. How much for 7 people?",
    "What happens to a solution's boiling point when a solute is dissolved?",
    "Outline the steps of binary search and state its time complexity.",
    "Compare mitosis and meiosis in terms of chromosome count and purpose.",
    "If 5 machines make 5 widgets in 5 minutes, how long for 100 machines "
    "to make 100 widgets?",
    "Write a short paragraph about the invention of the printing press.",
    "Explain what a race condition is and give one concrete example.",
    "Convert 45 degrees Celsius to Fahrenheit and show the calculation.",
    "What is the purpose of a confidence interval in an experiment?",
    "Describe the difference between TCP and UDP for a junior engineer.",
    "A shirt costs 80 after a 20% discount. What was the original price?",
    "Explain gradient descent without using any equations.",
    "List three reasons a distributed system might lose data during a partition.",
    "How does a diffusion language model differ from an autoregressive one?",
    "Write a regular expression matching a valid IPv4 address.",
    "What is the sum of the first 50 positive even integers?",
    "Explain why floating point arithmetic is not associative.",
    "Describe the water cycle and name each phase transition involved.",
    "A car travels 240 km on 16 litres. What is its consumption per 100 km?",
    "Explain the halting problem and why it is undecidable.",
    "Give three practical uses of a hash table with different key types.",
]


def load_prompts(
    n: int,
    dataset: Optional[str] = "gsm8k",
    seed: int = 0,
    cache_dir: Optional[str] = None,
) -> List[str]:
    """Return ``n`` calibration prompts.

    Tries the requested HF dataset first and falls back to the bundled list --
    a quantization run must not die because a cluster node has no internet.
    """
    rng = random.Random(seed)

    if dataset:
        try:
            texts = _load_hf(dataset, n, rng, cache_dir)
            if texts:
                return texts[:n]
        except Exception as exc:  # noqa: BLE001 - offline is a normal case here
            print(f"[calib] dataset '{dataset}' unavailable ({exc.__class__.__name__}: "
                  f"{exc}); using the bundled prompt set")

    pool = list(FALLBACK_PROMPTS)
    rng.shuffle(pool)
    while len(pool) < n:
        pool.extend(FALLBACK_PROMPTS)
    return pool[:n]


def _load_hf(name: str, n: int, rng: random.Random, cache_dir) -> List[str]:
    from datasets import load_dataset

    if name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="train", cache_dir=cache_dir)
        idx = rng.sample(range(len(ds)), min(n * 2, len(ds)))
        return [ds[i]["question"] for i in idx]

    if name in ("c4", "allenai/c4"):
        ds = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            cache_dir=cache_dir,
        )
        idx = rng.sample(range(min(len(ds), 20000)), min(n * 2, 20000))
        return [ds[i]["text"][:2000] for i in idx]

    ds = load_dataset(name, split="train", cache_dir=cache_dir)
    field = next(
        (f for f in ("text", "question", "prompt", "instruction") if f in ds.column_names),
        None,
    )
    if field is None:
        raise ValueError(f"no text-like column in {name}: {ds.column_names}")
    idx = rng.sample(range(len(ds)), min(n * 2, len(ds)))
    return [ds[i][field] for i in idx]


def mix_prompts(sources: Sequence[tuple[str, int]], seed: int = 0) -> List[str]:
    """Blend several datasets, e.g. [('gsm8k', 24), ('c4', 8)]."""
    out: List[str] = []
    for name, count in sources:
        out.extend(load_prompts(count, dataset=name, seed=seed))
    random.Random(seed).shuffle(out)
    return out


__all__ = ["load_prompts", "mix_prompts", "FALLBACK_PROMPTS"]
