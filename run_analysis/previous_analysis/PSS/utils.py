"""
analysis/utils.py — Reusable helper functions for the PSS analysis stage.

Embedding and entity logic now lives in the dedicated utils/ packages.
This module re-exports everything so that existing imports are unchanged:

    from analysis.utils import get_embeddings, extract_entities, ...

Public API
----------
load_json_outputs(path)           Load .jsonl or .json → list of record dicts.
get_embeddings(texts)             Encode texts with sentence-transformers.
avg_cosine_vs_base(embeddings)    Mean cosine sim of variants vs. base (row 0).
word_token_set(text)              Lowercase word-token set for a string.
lexical_overlap_vs_base(texts)    Mean word-token Jaccard of variants vs. base.
response_length_variance(texts)   Variance of word-token counts across texts.
ngram_set(text, n)                Word n-gram set (kept for compatibility).
jaccard(set_a, set_b)             Jaccard similarity between two sets.
extract_entities(text, language)  NER via spaCy or regex fallback.
primary_entity(entities)          First entity string (lowercased), or None.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Set

import numpy as np

# ---------------------------------------------------------------------------
# Re-export from dedicated utils packages
# ---------------------------------------------------------------------------
from utils.embedding_utils import (  # noqa: F401
    avg_cosine_vs_base,
    get_embed_model,
    get_embeddings,
)
from utils.entity_utils import (  # noqa: F401
    extract_entities,
    primary_entity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json_outputs(path: str) -> List[dict]:
    """
    Load a raw-outputs file and return a list of record dicts.

    Supports both formats automatically:
    • ``.jsonl`` — JSON Lines, one object per line (incremental format).
    • ``.json``  — legacy JSON array.
    """
    path = str(path)
    with open(path, "r", encoding="utf-8") as fh:
        if path.endswith(".jsonl"):
            data = [json.loads(line) for line in fh if line.strip()]
        else:
            raw = fh.read()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # File is JSON Lines written with a .json extension (incremental mode).
                data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    logger.info("Loaded %d records from %s", len(data), path)
    return data


# ---------------------------------------------------------------------------
# Lexical overlap — word-token Jaccard
# ---------------------------------------------------------------------------

def word_token_set(text: str) -> Set[str]:
    """Return the set of lowercase word tokens in ``text``."""
    return set(text.lower().split())


def jaccard(set_a: Set, set_b: Set) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|. Returns 1.0 for two empty sets."""
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def lexical_overlap_vs_base(texts: List[str]) -> float:
    """
    Average word-token Jaccard similarity of each variant response vs. base.

    texts[0] is the base; texts[1:] are the variants.
    Returns 1.0 when fewer than 2 texts are provided.
    """
    if len(texts) < 2:
        return 1.0
    base_tokens = word_token_set(texts[0])
    scores = [jaccard(base_tokens, word_token_set(t)) for t in texts[1:]]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Response length variance
# ---------------------------------------------------------------------------

def response_length_variance(texts: List[str]) -> float:
    """
    Population variance of word-token counts across all responses in a group.

    Returns 0.0 when len(texts) < 2.
    """
    if len(texts) < 2:
        return 0.0
    lengths = [len(t.split()) for t in texts]
    return float(np.var(lengths))


# ---------------------------------------------------------------------------
# n-gram helpers (kept for backward compatibility and tests)
# ---------------------------------------------------------------------------

def ngram_set(text: str, n: int = 3) -> Set[str]:
    """Return the set of space-joined word-level n-grams in ``text``."""
    tokens = text.lower().split()
    if len(tokens) < n:
        return set(tokens)
    return {" ".join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)}


def avg_lexical_vs_base(texts: List[str], n: int = 3) -> float:
    """Average word n-gram Jaccard vs. base (legacy; kept for tests)."""
    if len(texts) < 2:
        return 1.0
    base_grams = ngram_set(texts[0], n)
    scores = [jaccard(base_grams, ngram_set(t, n)) for t in texts[1:]]
    return float(np.mean(scores))
