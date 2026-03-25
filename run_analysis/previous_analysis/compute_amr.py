import json
import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.answer_utils import answer_match
from config.experiment_config import ANALYSIS_DIR, DATA_DIR

ANALYSIS_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=DATA_DIR / "raw_outputs.json",
        help="Path to raw_outputs.json (default: data/raw_outputs.json)"
    )
    return parser.parse_args()


def compute_amr(records: list[dict]) -> pd.DataFrame:

    rows = []
    for rec in records:
        rec = rec.copy()
        rec["amr"] = answer_match(rec.get("response"), rec.get("ground_truth"))
        rows.append(rec)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "language", "in_distribution"])
          .agg(
              mean_amr    = ("amr", "mean"),
              std_amr     = ("amr", "std"),
              n_prompts   = ("prompt_id", "nunique"),
              n_responses = ("amr", "count"),
          )
          .round(4)
          .reset_index()
          .sort_values(["model", "mean_amr"], ascending=[True, False])
    )


def flag_inconsistent(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["prompt_id", "model", "language"])["amr"]
          .mean()
          .reset_index()
          .query("amr < 0.3")
          .sort_values("amr")
    )



if __name__ == "__main__":
    args = parse_args()

    print(f"Loading raw outputs from {args.input}...")
    with open(args.input) as f:
        records = json.load(f)

    print(f"Computing AMR for {len(records)} records...")
    df = compute_amr(records)
    df.to_csv(ANALYSIS_DIR / "amr_results.csv", index=False)

    summary = summarise(df)
    summary.to_csv(ANALYSIS_DIR / "amr_scores.csv", index=False)

    inconsistent = flag_inconsistent(df)
    inconsistent.to_csv(ANALYSIS_DIR / "inconsistent_prompts.csv", index=False)

    # Console output
    print("\nMean AMR by model:")
    print(df.groupby("model")["amr"].mean().round(3).to_string())

    print("\nMean AMR — in-distribution vs out-of-distribution:")
    print(df.groupby(["model", "in_distribution"])["amr"].mean().round(3).to_string())

    print("\nMean AMR by model and language:")
    print(df.groupby(["model", "language"])["amr"].mean().round(3).unstack().to_string())

    print(f"\n{len(inconsistent)} low-AMR (prompt, model, language) combinations flagged.")
    print(f"Results saved to {ANALYSIS_DIR}/")