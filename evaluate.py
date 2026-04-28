"""QA-style metrics (token F1, exact match, token recall on gold)."""

from __future__ import annotations

import collections
import re
import string


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text, flags=re.IGNORECASE)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text.lower() if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s)))


def get_tokens(s: str) -> list[str]:
    if not s:
        return []
    return normalize_answer(s).split()


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(ground_truth)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    rec = num_same / len(gold_tokens)
    return (2 * precision * rec) / (precision + rec)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def recall(prediction: str, ground_truth: str) -> float:
    """Fraction of gold tokens (with multiplicity cap by prediction) covered by prediction."""
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(ground_truth)
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    return num_same / len(gold_tokens)
