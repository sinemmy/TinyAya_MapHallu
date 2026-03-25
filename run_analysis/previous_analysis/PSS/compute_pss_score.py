"""
analysis/compute_pss_score.py — Stage 3: Compute composite Prompt Sensitivity Score.

Reads previously computed per-metric results from analysis/pss_results.csv.
Does NOT call the model and does NOT recompute any metrics.

Composite PSS formula
---------------------
    prompt_sensitivity_score =
        (1 - semantic_similarity)
        * entity_change_rate
        * (1 - lexical_overlap)
        * normalized_response_length_variance

where:
    normalized_response_length_variance =
        response_length_variance / mean(response_length_variance)

Rows where entity_change_rate is null (Arabic / Hindi — non-Latin script,
regex fallback finds no entities) will have a null PSS score.  They are
excluded from the instability ranking but retained in the saved CSV.

Outputs
-------
analysis/pss_scores.csv       — pss_results.csv + prompt_sensitivity_score column
analysis/unstable_prompts.csv — top-20 most unstable prompts ranked by avg PSS

Usage
-----
    python analysis/compute_pss_score.py
    python analysis/compute_pss_score.py --input analysis/pss_results.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT      = "analysis/pss_results.csv"
DEFAULT_SCORES_OUT = "analysis/pss_scores.csv"
DEFAULT_UNSTABLE   = "analysis/unstable_prompts.csv"
TOP_N_UNSTABLE     = 20


# ---------------------------------------------------------------------------
# PSS computation
# ---------------------------------------------------------------------------

def add_pss_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``prompt_sensitivity_score`` column to ``df``.

    Steps
    -----
    1. Normalise ``response_length_variance`` by its column mean so that it
       contributes on a comparable scale to the other terms (which are all
       bounded in [0, 1]).
    2. Multiply the four terms.  Any row with a null ``entity_change_rate``
       produces a null PSS (propagated naturally by pandas).

    Parameters
    ----------
    df : pd.DataFrame  output of compute_pss.py

    Returns
    -------
    pd.DataFrame with an additional ``prompt_sensitivity_score`` column
    """
    df = df.copy()

    mean_var = df["response_length_variance"].mean()
    if mean_var == 0 or np.isnan(mean_var):
        logger.warning(
            "Mean response_length_variance is 0 or NaN — "
            "setting normalized_response_length_variance to 0."
        )
        df["normalized_rlv"] = 0.0
    else:
        df["normalized_rlv"] = df["response_length_variance"] / mean_var

    df["prompt_sensitivity_score"] = (
        (1 - df["semantic_similarity"])
        * df["entity_change_rate"]           # null for ar/hi → propagates null
        * (1 - df["lexical_overlap"])
        * df["normalized_rlv"]
    )

    # Drop the helper column from the saved output.
    df = df.drop(columns=["normalized_rlv"])
    return df


# ---------------------------------------------------------------------------
# Instability mapping
# ---------------------------------------------------------------------------

def build_unstable_table(df: pd.DataFrame, top_n: int = TOP_N_UNSTABLE) -> pd.DataFrame:
    """
    Rank prompts by their average ``prompt_sensitivity_score`` across languages.

    Null scores (ar/hi) are excluded from the average so the ranking reflects
    languages where the score could actually be computed.

    Parameters
    ----------
    df    : pd.DataFrame  output of add_pss_score()
    top_n : int           how many prompts to return

    Returns
    -------
    pd.DataFrame with columns:
        prompt_id, avg_prompt_sensitivity_score, languages_present
    """
    # Work only on rows where PSS is not null.
    scored = df.dropna(subset=["prompt_sensitivity_score"])

    grouped = scored.groupby("prompt_id").agg(
        avg_prompt_sensitivity_score=("prompt_sensitivity_score", "mean"),
        languages_present=("language", lambda s: ",".join(sorted(s.unique()))),
    ).reset_index()

    ranked = (
        grouped
        .sort_values("avg_prompt_sensitivity_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    ranked["avg_prompt_sensitivity_score"] = ranked[
        "avg_prompt_sensitivity_score"
    ].round(6)
    return ranked


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, unstable: pd.DataFrame) -> None:
    """
    Print a clean three-section summary to stdout.

    Sections
    --------
    1. Mean PSS overall
    2. Mean PSS by language
    3. Top-10 most unstable prompts
    """
    scored = df.dropna(subset=["prompt_sensitivity_score"])

    print()
    print("=" * 60)
    print("  Prompt Sensitivity Score — Summary")
    print("=" * 60)

    # 1. Overall mean
    overall_mean = scored["prompt_sensitivity_score"].mean()
    overall_std  = scored["prompt_sensitivity_score"].std()
    print(f"\n  Overall mean PSS : {overall_mean:.6f}  (std {overall_std:.6f})")
    print(f"  Scored rows      : {len(scored)} / {len(df)} total")
    print(f"  Null PSS rows    : {df['prompt_sensitivity_score'].isna().sum()}"
          "  (ar/hi — no entity extraction)")

    # 2. By language
    print()
    print("  Mean PSS by language")
    print("  " + "-" * 40)
    lang_means = (
        scored.groupby("language")["prompt_sensitivity_score"]
        .agg(mean="mean", std="std", count="count")
        .round(6)
    )
    print(lang_means.to_string())

    # 3. Top-10 most unstable
    print()
    print("  Top-10 most unstable prompts")
    print("  " + "-" * 40)
    print(unstable.head(10).to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute composite PSS from pre-computed metric CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",   default=DEFAULT_INPUT,
                        help="Path to pss_results.csv")
    parser.add_argument("--scores",  default=DEFAULT_SCORES_OUT,
                        help="Output path for pss_scores.csv")
    parser.add_argument("--unstable", default=DEFAULT_UNSTABLE,
                        help="Output path for unstable_prompts.csv")
    parser.add_argument("--top_n",  type=int, default=TOP_N_UNSTABLE,
                        help="Number of most-unstable prompts to report")
    return parser.parse_args()


def _setup_file_logging() -> None:
    from datetime import datetime
    from pathlib import Path as _Path
    _Path("logs").mkdir(exist_ok=True)
    fh = logging.FileHandler(
        f"logs/compute_pss_score_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)


def main() -> None:
    args = _parse_args()
    _setup_file_logging()
    logger.info("Starting analysis stage — compute_pss_score.py")

    # Load
    df = pd.read_csv(args.input)
    logger.info("Loaded %d rows from %s", len(df), args.input)

    # Compute composite score
    df = add_pss_score(df)
    logger.info(
        "PSS computed.  Non-null scores: %d / %d",
        df["prompt_sensitivity_score"].notna().sum(), len(df),
    )

    # Save scores
    Path(args.scores).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.scores, index=False, encoding="utf-8")
    logger.info("Scores saved  →  %s", args.scores)

    # Instability table
    unstable = build_unstable_table(df, top_n=args.top_n)
    unstable.to_csv(args.unstable, index=False, encoding="utf-8")
    logger.info("Unstable prompts saved  →  %s", args.unstable)

    # Prompt instability dataset — all prompts ranked by avg PSS, with prompt_text
    instability_path = Path(args.scores).parent / "prompt_instability.csv"
    scored = df.dropna(subset=["prompt_sensitivity_score"])
    pt_col = "prompt_text" if "prompt_text" in scored.columns else None
    agg_dict = {"avg_pss_across_languages": ("prompt_sensitivity_score", "mean")}
    if pt_col:
        agg_dict["prompt_text"] = (pt_col, "first")
    instability = (
        scored.groupby("prompt_id")
        .agg(**agg_dict)
        .reset_index()
        .sort_values("avg_pss_across_languages", ascending=False)
        .reset_index(drop=True)
    )
    instability["avg_pss_across_languages"] = instability["avg_pss_across_languages"].round(6)
    instability.to_csv(instability_path, index=False, encoding="utf-8")
    logger.info("Prompt instability saved  →  %s", instability_path)

    # Console summary
    print_summary(df, unstable)


if __name__ == "__main__":
    main()
