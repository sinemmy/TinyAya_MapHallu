"""
data/load_mkqa.py — Load MKQA dataset in standardized format.

Adapted from hallucination-rate/data/load_mkqa.py.
Returns a flat list of sample dicts with consistent schema.
"""

import gzip
import json
import random
import urllib.request
from typing import List

try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False

MKQA_URL = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"
_EXCLUDED_ANSWER_TYPES = {"unanswerable"}


def _has_valid_gold_answer(ans: list) -> bool:
    """Return True if the answer list contains at least one usable gold string."""
    if not isinstance(ans, list) or not ans:
        return False
    for a in ans:
        if isinstance(a, dict) and (a.get("type") or "").strip() in _EXCLUDED_ANSWER_TYPES:
            return False
    for a in ans:
        if not isinstance(a, dict):
            continue
        t = a.get("text")
        if isinstance(t, str) and t.strip():
            return True
        for alias in (a.get("aliases") or []):
            if isinstance(alias, str) and alias.strip():
                return True
    return False


def _load_mkqa_raw(max_examples: int | None = None) -> list[dict]:
    """Load full MKQA rows. Tries HuggingFace first, falls back to direct download."""
    rows: list[dict] = []
    if _HAS_DATASETS:
        try:
            ds = load_dataset("apple/mkqa", split="train", trust_remote_code=True)
            for i, row in enumerate(ds):
                if max_examples is not None and i >= max_examples:
                    break
                rows.append(dict(row))
            if rows:
                return rows
        except Exception:
            pass
    with urllib.request.urlopen(MKQA_URL) as resp:
        with gzip.open(resp, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if max_examples is not None and len(rows) >= max_examples:
                    break
    return rows


def load_mkqa(languages: list[str], num_samples: int | None = 500, seed: int = 42) -> list[dict]:
    """
    Load MKQA samples for the given languages.

    If num_samples is None, all samples are used.

    Returns list of dicts:
        {
            "sample_id": int,         # index within the sampled set
            "language": str,
            "prompt_fields": {"query": str},
            "gold_answers": list,     # MKQA answer dicts with text/aliases
        }

    Samples are aligned by sample_id across languages.
    """
    raw = _load_mkqa_raw()
    random.seed(seed)
    if num_samples is None:
        sampled = raw
    else:
        sampled = random.sample(raw, min(num_samples, len(raw)))

    rows = []
    for idx, s in enumerate(sampled):
        queries = s.get("queries") or {}
        answers = s.get("answers") or {}
        for lang in languages:
            q = (queries.get(lang) or queries.get("en") or "").strip()
            if not q:
                continue
            ans = answers.get(lang) or answers.get("en") or []
            if not _has_valid_gold_answer(ans):
                continue
            rows.append({
                "sample_id": idx,
                "language": lang,
                "prompt_fields": {"query": q},
                "gold_answers": ans,
            })
    return rows
