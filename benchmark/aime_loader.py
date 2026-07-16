"""
AIME dataset loading and answer scoring, reused verbatim from
early_detection/generate.py (which itself reused it from
token-efficiency-math-reasoning) so this project's 200-problem benchmark
set is IDENTICAL, in order and content, to the one the probe was validated
against. A different sample (even same dataset, different seed/ordering)
would make "the probe scores 0.612 AUC on these problems" and "here's how
it does when used for routing on these problems" refer to two different
populations, quietly undermining the comparison.
"""

import re

from datasets import load_dataset

from src.config import settings


def load_aime_samples() -> list[dict]:
    ds = load_dataset(settings.aime_dataset, split="train")
    ds = ds.shuffle(seed=settings.aime_seed).select(range(min(settings.aime_n_samples, len(ds))))
    return [{"question": r["Question"], "answer": str(r["Answer"])} for r in ds]


def extract_answer(response: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    if boxed:
        return boxed[-1].strip()
    ans = re.findall(r"[Tt]he answer is\s*\$?([0-9\-\/\.\,]+)", response)
    if ans:
        return ans[-1].strip()
    post = response.split("</think>")[-1] if "</think>" in response else response
    nums = re.findall(r"\$?([0-9]+(?:\.[0-9]+)?)", post)
    return nums[-1].strip() if nums else None


def normalize_answer(ans: str | None) -> str | None:
    if ans is None:
        return None
    ans = ans.replace(",", "").replace("$", "").strip()
    try:
        return str(float(ans))
    except ValueError:
        return ans.lower().strip()
