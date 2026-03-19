
import sys
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.experiment_config import ANALYSIS_DIR, PLOTS_DIR

PLOTS_DIR.mkdir(exist_ok=True)

MODEL_PALETTE = {
    "global": "#5B8DB8",
    "earth":  "#E07B54",
    "water":  "#6BAE8E",
}

sns.set_theme(style="whitegrid", font_scale=1.1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amr", type=Path, default=ANALYSIS_DIR / "amr_results.csv")
    parser.add_argument("--scs", type=Path, default=ANALYSIS_DIR / "scs_results.csv")
    parser.add_argument("--clc", type=Path, default=ANALYSIS_DIR / "clc_scores.csv")
    return parser.parse_args()


# ── Plot 1 ─────────────────────────────────────────────────────────────────────

def plot_amr_by_language(amr_df: pd.DataFrame):
    agg = (
        amr_df.groupby(["model", "language"])["amr"]
              .mean()
              .reset_index()
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(
        data=agg, x="language", y="amr", hue="model",
        palette=MODEL_PALETTE, ax=ax
    )
    ax.set_title("Answer Match Rate by Language and Model", fontweight="bold", pad=12)
    ax.set_ylabel("Mean AMR")
    ax.set_xlabel("Language")
    ax.set_ylim(0, 0.5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.axhline(0.2, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "clc_by_language.png", dpi=150)
    plt.close()
    print("Saved: clc_by_language.png")


# ── Plot 2 ─────────────────────────────────────────────────────────────────────

def plot_indist_vs_outdist(amr_df: pd.DataFrame):
    agg = (
        amr_df.groupby(["model", "in_distribution"])["amr"]
              .mean()
              .reset_index()
    )
    agg["distribution"] = agg["in_distribution"].map({
        True:  "In-distribution",
        False: "Out-of-distribution",
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=agg, x="model", y="amr", hue="distribution",
        palette={"In-distribution": "#E07B54", "Out-of-distribution": "#5B8DB8"},
        ax=ax,
    )
    ax.set_title(
        "AMR: In-Distribution vs Out-of-Distribution Languages",
        fontweight="bold", pad=12,
    )
    ax.set_ylabel("Mean AMR")
    ax.set_xlabel("Model Variant")
    ax.set_ylim(0, 0.35)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(title="Language type", bbox_to_anchor=(1.01, 1), loc="upper left")

    # Annotate Earth's reversal
    ax.annotate(
        "Earth scores lower\non its own languages",
        xy=(0, 0.177), xytext=(0.35, 0.27),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=9, color="black",
    )

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "indist_vs_outdist.png", dpi=150)
    plt.close()
    print("Saved: indist_vs_outdist.png")


# ── Plot 3 ─────────────────────────────────────────────────────────────────────

def plot_scs_distribution(scs_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in scs_df.groupby("model"):
        ax.hist(
            group["scs"].dropna(), bins=15, alpha=0.6,
            label=model, color=MODEL_PALETTE.get(model, "gray"),
            edgecolor="white", linewidth=0.5,
        )
    ax.set_title(
        "Semantic Consistency Score Distribution by Model",
        fontweight="bold", pad=12,
    )
    ax.set_xlabel("SCS (mean pairwise cosine similarity across languages)")
    ax.set_ylabel("Number of prompts")
    ax.legend(title="Model")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "clc_distribution.png", dpi=150)
    plt.close()
    print("Saved: clc_distribution.png")


# ── Plot 4 ─────────────────────────────────────────────────────────────────────

def plot_global_vs_regional(clc_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=clc_df, x="model", y="clc_score",
        palette=MODEL_PALETTE, ax=ax,
        order=["global", "earth", "water"],
        width=0.5,
    )
    ax.set_title(
        "CLC Score Distribution: Global vs Regional Aya Variants",
        fontweight="bold", pad=12,
    )
    ax.set_ylabel("CLC Score (AMR×0.6 + SCS×0.4)")
    ax.set_xlabel("Model Variant")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "global_vs_regional.png", dpi=150)
    plt.close()
    print("Saved: global_vs_regional.png")


# ── Plot 5 ─────────────────────────────────────────────────────────────────────

def plot_amr_heatmap(amr_df: pd.DataFrame):
    pivot = (
        amr_df.groupby(["model", "language"])["amr"]
              .mean()
              .unstack()
              .round(3)
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0, vmax=0.4, ax=ax,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Mean AMR"},
    )
    ax.set_title(
        "Mean AMR Heatmap — Model × Language",
        fontweight="bold", pad=12,
    )
    ax.set_ylabel("Model")
    ax.set_xlabel("Language")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "amr_heatmap.png", dpi=150)
    plt.close()
    print("Saved: amr_heatmap.png")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    print("Loading results...")
    amr_df = pd.read_csv(args.amr)
    scs_df = pd.read_csv(args.scs)
    clc_df = pd.read_csv(args.clc)

    print("Generating plots...\n")
    plot_amr_by_language(amr_df)
    plot_indist_vs_outdist(amr_df)
    plot_scs_distribution(scs_df)
    plot_global_vs_regional(clc_df)
    plot_amr_heatmap(amr_df)

    print(f"\nAll plots saved to {PLOTS_DIR}/")