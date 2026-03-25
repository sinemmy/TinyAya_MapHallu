
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, UTC

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.experiment_config import ANALYSIS_DIR

ANALYSIS_DIR.mkdir(exist_ok=True)
AMR_WEIGHT = 0.6
SCS_WEIGHT = 0.4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--amr", type=Path, default=ANALYSIS_DIR / "amr_results.csv",
        help="Path to amr_results.csv (default: analysis/amr_results.csv)"
    )
    parser.add_argument(
        "--scs", type=Path, default=ANALYSIS_DIR / "scs_results.csv",
        help="Path to scs_results.csv (default: analysis/scs_results.csv)"
    )
    return parser.parse_args()


def merge_and_score(amr_df: pd.DataFrame, scs_df: pd.DataFrame) -> pd.DataFrame:
    amr_agg = (
        amr_df.groupby(["prompt_id", "model"])["amr"]
              .mean()
              .reset_index()
              .rename(columns={"amr": "mean_amr"})
    )
    merged = amr_agg.merge(
        scs_df[["prompt_id", "model", "scs"]],
        on=["prompt_id", "model"],
        how="inner",
    )
    merged["clc_score"] = (
        AMR_WEIGHT * merged["mean_amr"] + SCS_WEIGHT * merged["scs"]
    ).round(4)
    return merged


def write_summary(clc_df: pd.DataFrame, amr_df: pd.DataFrame):
    lines = [
        "Cross-Language Consistency — Summary Report",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "=" * 60,
        "",
        "CLC Score by Model (AMR×0.6 + SCS×0.4):",
        clc_df.groupby("model")["clc_score"].mean().round(3).to_string(),
        "",
        "Mean AMR by Model:",
        amr_df.groupby("model")["amr"].mean().round(3).to_string(),
        "",
        "AMR — In-Distribution vs Out-of-Distribution:",
        amr_df.groupby(["model", "in_distribution"])["amr"].mean().round(3).to_string(),
        "",
        "Mean SCS by Model:",
        clc_df.groupby("model")["scs"].mean().round(3).to_string(),
        "",
        "=" * 60,
        "Redistribution flag — models where in-dist AMR exceeds out-of-dist by >0.1:",
    ]
    pivot = (
        amr_df.groupby(["model", "in_distribution"])["amr"]
              .mean()
              .unstack()
    )
    if True in pivot.columns and False in pivot.columns:
        pivot.columns = ["out_dist", "in_dist"]
        pivot["gap"] = pivot["in_dist"] - pivot["out_dist"]
        flagged = pivot[pivot["gap"] > 0.1]
        if len(flagged):
            lines.append(flagged.round(3).to_string())
            lines.append(
                "\n^ These models improved in-distribution at cost to other languages."
            )
        else:
            lines.append("  None detected at >0.1 threshold.")
    else:
        lines.append("  Could not compute — missing in/out distribution labels.")

    summary_path = ANALYSIS_DIR / "clc_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading AMR results from {args.amr}...")
    amr_df = pd.read_csv(args.amr)

    print(f"Loading SCS results from {args.scs}...")
    scs_df = pd.read_csv(args.scs)

    print("Merging and computing CLC scores...")
    clc_df = merge_and_score(amr_df, scs_df)
    clc_df.to_csv(ANALYSIS_DIR / "clc_scores.csv", index=False)

    print("\nCLC Score by Model:")
    print(clc_df.groupby("model")["clc_score"].mean().round(3).to_string())

    write_summary(clc_df, amr_df)
    print(f"\nAll results saved to {ANALYSIS_DIR}/")