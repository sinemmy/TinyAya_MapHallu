

import json
import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.embedding_utils import compute_scs
from config.experiment_config import ANALYSIS_DIR, DATA_DIR

ANALYSIS_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=DATA_DIR / "raw_outputs.json",
        help="Path to raw_outputs.json (default: data/raw_outputs.json)"
    )
    return parser.parse_args()


def compute_scs_scores(df: pd.DataFrame) -> pd.DataFrame:
    df0 = df[df["sample_idx"] == 0].copy()

    results = []
    groups = list(df0.groupby(["prompt_id", "model"]))
    total = len(groups)

    for i, ((prompt_id, model), group) in enumerate(groups):
        responses = group["response"].tolist()
        langs     = group["language"].tolist()
        scs       = compute_scs(responses)

        print(f"  [{i+1}/{total}] {model} | {prompt_id} | SCS={scs}")
        results.append({
            "prompt_id":   prompt_id,
            "model":       model,
            "n_languages": len(responses),
            "languages":   ",".join(langs),
            "scs":         scs,
        })

    return pd.DataFrame(results)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("model")
          .agg(
              mean_scs = ("scs", "mean"),
              std_scs  = ("scs", "std"),
              n        = ("prompt_id", "count"),
          )
          .round(4)
          .reset_index()
          .sort_values("mean_scs", ascending=False)
    )


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading raw outputs from {args.input}...")
    with open(args.input) as f:
        records = json.load(f)

    df = pd.DataFrame(records)

    print(f"Computing SCS across {df['prompt_id'].nunique()} prompts "
          f"and {df['model'].nunique()} models...\n"
          f"(embedding model loads on first call — takes ~10s)\n")

    scs_df = compute_scs_scores(df)
    scs_df.to_csv(ANALYSIS_DIR / "scs_results.csv", index=False)

    summary = summarise(scs_df)
    summary.to_csv(ANALYSIS_DIR / "scs_scores.csv", index=False)

    print("\nMean SCS by model:")
    print(summary.to_string(index=False))

    print(f"\nResults saved to {ANALYSIS_DIR}/")