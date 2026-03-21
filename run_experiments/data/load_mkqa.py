"""
data/load_mkqa.py — Load MKQA dataset in standardized format.

Adapted from hallucination-rate/data/load_mkqa.py.
Returns a flat list of sample dicts with consistent schema.
"""

import gzip
import json
import logging
import os
import random
import urllib.request
from typing import List

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False

MKQA_URL = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"
_EXCLUDED_ANSWER_TYPES = {"unanswerable"}

MKQA_LANGUAGES = {
    "ar", "da", "de", "en", "es", "fi", "fr", "he", "hu", "it", "ja",
    "km", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "th", "tr",
    "vi", "zh_cn", "zh_hk", "zh_tw",
}

# Common aliases that map to actual MKQA language keys
_LANGUAGE_ALIASES = {"zh": "zh_cn"}


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
            ds = load_dataset("apple/mkqa", split="train", trust_remote_code=True, token=os.getenv("HF_TOKEN"))
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

    Samples num_samples rows **per language** from the rows that have valid data
    for that language, so every language always gets exactly num_samples records
    (or as many as exist). If num_samples is None, all valid rows are used.

    Returns list of dicts:
        {
            "sample_id": int,         # index in the raw MKQA dataset
            "language": str,
            "prompt_fields": {"query": str},
            "gold_answers": list,     # MKQA answer dicts with text/aliases
        }
    """
    raw = _load_mkqa_raw()
    random.seed(seed)

    # Resolve aliases (e.g. zh -> zh_cn) before validation
    resolved = [_LANGUAGE_ALIASES.get(l, l) for l in languages]
    for orig, res in zip(languages, resolved):
        if orig != res:
            logger.info("MKQA: mapping '%s' -> '%s'", orig, res)

    unsupported = [l for l in resolved if l not in MKQA_LANGUAGES]
    if unsupported:
        logger.warning("Skipping languages not available in MKQA: %s. Valid: %s", unsupported, sorted(MKQA_LANGUAGES))

    # Build mapping from MKQA internal key back to the caller's original code
    mkqa_to_orig = {}
    for orig, res in zip(languages, resolved):
        if res in MKQA_LANGUAGES:
            mkqa_to_orig[res] = orig
    valid_languages = list(mkqa_to_orig.keys())

    rows = []
    for lang in valid_languages:
        # Collect all raw indices that have valid data for this language
        valid_indices = [
            i for i, s in enumerate(raw)
            if (s.get("queries") or {}).get(lang, "").strip()
            and _has_valid_gold_answer((s.get("answers") or {}).get(lang) or [])
        ]

        if num_samples is None:
            selected = valid_indices
        else:
            selected = random.sample(valid_indices, min(num_samples, len(valid_indices)))

        for idx in selected:
            s = raw[idx]
            rows.append({
                "sample_id": idx,
                "language": mkqa_to_orig[lang],
                "prompt_fields": {"query": s["queries"][lang].strip()},
                "gold_answers": s["answers"][lang],
            })

    return rows
