"""
data/load_xnli.py — Load XNLI dataset in standardized format.

Adapted from CMDR/CMDR.py load_multilingual_data().
Returns a flat list of sample dicts with consistent schema.
"""

from datasets import load_dataset

# XNLI integer label → string
_LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}


def load_xnli(languages: list[str], num_samples: int | None = 300, seed: int = 42) -> list[dict]:
    """
    Load aligned XNLI samples for each language.

    If num_samples is None, all samples are used.

    Returns list of dicts:
        {
            "sample_id": int,
            "language": str,
            "prompt_fields": {"premise": str, "hypothesis": str},
            "gold_label": str,       # "entailment" | "neutral" | "contradiction"
        }

    Samples are aligned by index across languages (XNLI guarantees this).
    """
    rows = []
    for lang in languages:
        ds = load_dataset("xnli", lang, split="test")
        n = len(ds) if num_samples is None else min(num_samples, len(ds))
        for i in range(n):
            rows.append({
                "sample_id": i,
                "language": lang,
                "prompt_fields": {
                    "premise": ds[i]["premise"],
                    "hypothesis": ds[i]["hypothesis"],
                },
                "gold_label": _LABEL_MAP.get(ds[i]["label"], str(ds[i]["label"])),
            })
    return rows
