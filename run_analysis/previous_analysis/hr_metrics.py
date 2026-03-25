"""
Hallucination rate from MKQA gold answers.

Correct = model response matches (or contains) at least one acceptable gold answer.
Hallucination rate = 1 - (number correct / total).

is_correct: DONE (included in output/*.json in run_experiments.py) 
"""

import re
from typing import Any, List


def _normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    if not s or not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.lower().strip())


def get_acceptable_answer_strings(answers: List[dict]) -> List[str]:
    """
    Extract all acceptable answer strings from MKQA answers list.

    Each item can have "text" and "aliases" (list). Returns normalized non-empty strings.
    """
    out = []
    for a in answers or []:
        if isinstance(a, dict):
            t = a.get("text")
            if t and isinstance(t, str) and t.strip():
                out.append(_normalize(t))
            for alias in a.get("aliases") or []:
                if alias and isinstance(alias, str) and alias.strip():
                    out.append(_normalize(alias))
    return list(dict.fromkeys(out))  # dedupe preserving order


def is_correct(response_text: str, answers: List[dict]) -> bool:
    """
    True if the model response is considered correct given MKQA gold answers.

    Uses containment: response (normalized) must contain at least one acceptable gold string.
    Empty response or no gold answers -> False.
    """
    gold_strings = get_acceptable_answer_strings(answers)
    if not gold_strings:
        return False
    resp_norm = _normalize(response_text)
    if not resp_norm:
        return False
    return any(g in resp_norm for g in gold_strings)


def compute_hallucination_rate(results: List[dict]) -> dict:
    """
    From evaluation results (each with "response_text" and "answers"), compute rates.

    Returns dict with:
      - hallucination_rate: float in [0, 1]
      - accuracy: float in [0, 1]
      - n_correct: int
      - n_total: int
      - n_no_gold: int (items with no gold answers, excluded from rate)
    """
    n_correct = 0
    n_total = 0
    n_no_gold = 0
    for r in results:
        answers = r.get("answers")
        gold_strings = get_acceptable_answer_strings(answers if isinstance(answers, list) else [])
        if not gold_strings:
            n_no_gold += 1
            continue
        n_total += 1
        if is_correct(r.get("response_text", ""), answers if isinstance(answers, list) else []):
            n_correct += 1
    if n_total == 0:
        return {
            "hallucination_rate": 0.0,
            "accuracy": 0.0,
            "n_correct": 0,
            "n_total": 0,
            "n_no_gold": n_no_gold,
        }
    accuracy = n_correct / n_total
    return {
        "hallucination_rate": 1.0 - accuracy,
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": n_total,
        "n_no_gold": n_no_gold,
    }


def add_correctness_to_results(results: List[dict]) -> List[dict]:
    """Add "is_correct" to each result record. Mutates and returns the same list."""
    for r in results:
        answers = r.get("answers")
        if not isinstance(answers, list):
            r["is_correct"] = False
            continue
        r["is_correct"] = is_correct(r.get("response_text", ""), answers)
    return results
