"""
analysis/compute_pss.py — Stage 2: Compute PSS metrics from raw model outputs.

This script does NOT call any model or API.
It reads previously saved JSONL outputs and computes four sensitivity signals.

Metrics per (prompt_id, language)
----------------------------------
semantic_similarity       Mean cosine similarity of variant embeddings vs. base.
                          Uses sentence-transformers/all-MiniLM-L6-v2.
                          Base-to-base is excluded from the mean.

entity_change_rate        Fraction of variants whose primary named entity differs
                          from the base response's entity.  Null when extraction
                          fails for the base response.

lexical_overlap           Mean word-token Jaccard similarity of each variant
                          response vs. the base response.

response_length_variance  Population variance of word-token counts across all
                          responses in the group (base + 4 variants).

Outputs
-------
analysis/pss_results.csv   — one row per (prompt_id, language)
analysis/pss_summary.txt   — human-readable summary statistics

Usage
-----
    python analysis/compute_pss.py \\
        --input  data/raw_outputs.jsonl \\
        --output analysis/pss_results.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project root is on sys.path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from PSS.utils import (  # noqa: E402
    avg_cosine_vs_base,
    extract_entities,
    get_embeddings,
    lexical_overlap_vs_base,
    load_json_outputs,
    primary_entity,
    response_length_variance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Metric columns in the output CSV (order matters for display).
METRIC_COLS = [
    "semantic_similarity",
    "entity_change_rate",
    "lexical_overlap",
    "response_length_variance",
]


# ---------------------------------------------------------------------------
# Per-group metric computation
# ---------------------------------------------------------------------------

def compute_pss_for_group(group: pd.DataFrame) -> dict:
    """
    Compute all four PSS signals for one (prompt_id, language) group.

    The row with variant_type == 'base' is sorted first so every metric is
    measured relative to the base response.

    Parameters
    ----------
    group : pd.DataFrame  rows for a single (prompt_id, language) pair

    Returns
    -------
    dict with keys matching METRIC_COLS; values are float or None
    """
    # Sort so the 'base' row is always at index 0.
    group = group.sort_values(
        "variant_type",
        key=lambda s: s.map(lambda x: 0 if x == "base" else 1),
    ).reset_index(drop=True)

    responses: list = group["response"].fillna("").tolist()
    language:  str  = group["language"].iloc[0]
    pid:       int  = group["prompt_id"].iloc[0]

    # ------------------------------------------------------------------
    # 1. Semantic similarity  (embedding cosine vs. base)
    # ------------------------------------------------------------------
    if all(r == "" for r in responses):
        sem_sim = None
        logger.warning("All responses empty — skipping embeddings (pid=%s, lang=%s).", pid, language)
    else:
        safe = [r if r.strip() else " " for r in responses]
        embeddings = get_embeddings(safe)
        sem_sim    = avg_cosine_vs_base(embeddings)   # excludes base-to-base

    # ------------------------------------------------------------------
    # 2. Entity change rate
    # ------------------------------------------------------------------
    base_resp = responses[0] if responses else ""
    base_ent  = primary_entity(extract_entities(base_resp, language))

    if base_ent is None:
        entity_rate = None
        # Arabic and Hindi use non-Latin scripts — the regex fallback returns
        # nothing for them by design.  Log at DEBUG to avoid console spam;
        # the null count is reported in the summary statistics.
        logger.debug(
            "Base entity extraction failed — entity_change_rate=null (pid=%s, lang=%s).",
            pid, language,
        )
    else:
        n_changes, n_valid = 0, 0
        for resp in responses[1:]:   # variants only
            ent = primary_entity(extract_entities(resp, language))
            if ent is None:
                continue
            n_valid += 1
            if ent != base_ent:
                n_changes += 1
        entity_rate = (n_changes / n_valid) if n_valid > 0 else None

    # ------------------------------------------------------------------
    # 3. Lexical overlap  (word-token Jaccard vs. base)
    # ------------------------------------------------------------------
    lex_overlap = lexical_overlap_vs_base(responses)

    # ------------------------------------------------------------------
    # 4. Response length variance
    # ------------------------------------------------------------------
    len_variance = response_length_variance(responses)

    return {
        "semantic_similarity":      sem_sim,
        "entity_change_rate":       entity_rate,
        "lexical_overlap":          lex_overlap,
        "response_length_variance": len_variance,
    }


# ---------------------------------------------------------------------------
# Main computation loop
# ---------------------------------------------------------------------------

def compute_pss(records: list) -> pd.DataFrame:
    """
    Run PSS computation for every (prompt_id, language) group.

    Parameters
    ----------
    records : list of dicts from load_json_outputs

    Returns
    -------
    pd.DataFrame with columns: prompt_id, language, + METRIC_COLS
    """
    df     = pd.DataFrame(records)
    groups = list(df.groupby(["prompt_id", "language"]))
    logger.info("Computing PSS for %d groups …", len(groups))

    rows = []
    for (pid, lang), group in groups:
        metrics = compute_pss_for_group(group.copy())

        # Extract base response and prompt text for debugging / interpretation.
        base_row   = group[group["variant_type"] == "base"]
        base_resp  = base_row["response"].iloc[0] if not base_row.empty else ""
        # Support both old schema (base_prompt) and new schema (prompt_text).
        pt_col     = "prompt_text" if "prompt_text" in group.columns else "base_prompt"
        prompt_txt = base_row[pt_col].iloc[0] if not base_row.empty else ""

        rows.append({
            "prompt_id":    pid,
            "language":     lang,
            "prompt_text":  prompt_txt,
            "base_response": base_resp,
            **metrics,
        })
        logger.debug("pid=%s lang=%s done.", pid, lang)

    result_df = pd.DataFrame(rows)
    logger.info("Done. Result shape: %s", result_df.shape)
    return result_df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def build_summary(records: list, result_df: pd.DataFrame) -> str:
    """
    Build a human-readable summary string with four sections:

    1. Overall metric averages
    2. Per-language metric averages
    3. Top-10 most unstable prompts (lowest semantic_similarity)
    4. Dataset statistics

    Parameters
    ----------
    records   : raw records list (for dataset stats)
    result_df : output of compute_pss()

    Returns
    -------
    str — formatted report ready to print or write to a file
    """
    lines: list = []

    def header(title: str) -> None:
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"  {title}")
        lines.append("=" * 60)

    # ------------------------------------------------------------------
    # 1. Overall averages
    # ------------------------------------------------------------------
    header("1. Overall Metric Averages")
    for col in METRIC_COLS:
        vals = result_df[col].dropna()
        if vals.empty:
            lines.append(f"  {col:<30}  n/a")
        else:
            lines.append(
                f"  {col:<30}  mean={vals.mean():.4f}  "
                f"std={vals.std():.4f}  "
                f"[{vals.min():.4f}, {vals.max():.4f}]"
            )

    # ------------------------------------------------------------------
    # 2. Per-language averages
    # ------------------------------------------------------------------
    header("2. Metric Averages by Language")
    lang_stats = (
        result_df.groupby("language")[METRIC_COLS]
        .mean()
        .round(4)
    )
    lines.append(lang_stats.to_string())

    # ------------------------------------------------------------------
    # 3. Top-10 most unstable prompts
    # ------------------------------------------------------------------
    header("3. Top-10 Most Unstable Prompts  (lowest semantic_similarity)")
    ranked = (
        result_df.dropna(subset=["semantic_similarity"])
        .nsmallest(10, "semantic_similarity")
        [["prompt_id", "language", "semantic_similarity",
          "entity_change_rate", "lexical_overlap"]]
        .reset_index(drop=True)
    )
    if ranked.empty:
        lines.append("  No data available.")
    else:
        lines.append(ranked.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Dataset statistics
    # ------------------------------------------------------------------
    header("4. Dataset Statistics")
    raw_df        = pd.DataFrame(records)
    total_prompts = result_df["prompt_id"].nunique()
    total_responses = len(raw_df)
    languages     = sorted(raw_df["language"].unique().tolist())
    variants      = sorted(raw_df["variant_type"].unique().tolist())
    lines += [
        f"  Total unique prompts    : {total_prompts}",
        f"  Total responses         : {total_responses}",
        f"  Languages               : {', '.join(languages)}",
        f"  Variant types           : {', '.join(variants)}",
        f"  Groups (prompt × lang)  : {len(result_df)}",
        f"  Null entity_change_rate : {result_df['entity_change_rate'].isna().sum()}",
    ]

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PSS metrics from saved model outputs (no model calls).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="data/raw_outputs.json",
        help="Path to raw_outputs.json produced by collect_data.py.",
    )
    parser.add_argument(
        "--output",
        default="analysis/pss_results.csv",
        help="Destination CSV for per-group PSS results.",
    )
    parser.add_argument(
        "--summary",
        default="analysis/pss_summary.txt",
        help="Destination text file for human-readable summary statistics.",
    )
    return parser.parse_args()


def _setup_file_logging() -> None:
    from datetime import datetime
    from pathlib import Path as _Path
    _Path("logs").mkdir(exist_ok=True)
    fh = logging.FileHandler(
        f"logs/compute_pss_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)


def main() -> None:
    args = _parse_args()
    _setup_file_logging()
    logger.info("Starting analysis stage — compute_pss.py")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # --- Load & compute ---
    records   = load_json_outputs(args.input)

    # One-time notice: entity extraction is null for Arabic/Hindi by design
    # (non-Latin scripts; regex fallback finds no capitalised tokens).
    # Install spaCy for better English coverage:
    #   pip install spacy && python -m spacy download en_core_web_sm
    logger.info(
        "Note: entity_change_rate will be null for Arabic (ar) and Hindi (hi) "
        "responses — non-Latin scripts are not covered by the regex fallback."
    )

    result_df = compute_pss(records)

    # --- Save CSV ---
    result_df.to_csv(args.output, index=False, encoding="utf-8")
    logger.info("PSS results saved  →  %s", args.output)

    # --- Build summary ---
    summary = build_summary(records, result_df)

    # Print to console
    print(summary)

    # Save to text file
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as fh:
        fh.write(summary)
    logger.info("Summary saved      →  %s", args.summary)


if __name__ == "__main__":
    main()
