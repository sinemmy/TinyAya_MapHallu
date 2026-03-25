"""
Compute Semantic Consistency Score (SCS) for MKQA base responses.

For each (sample_id, model), embeds response_text across languages using a
multilingual sentence-transformer, then computes mean pairwise cosine similarity.

Saves results to results/scs_results.csv.

Usage:
    python compute_scs.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from load_data import load_all

DATA_ROOT = Path(__file__).parent / "../run_experiments/output"
SKIP_DIRS = {"water_base_20260322_051223"}
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 256


def short_model(name: str) -> str:
    return name.replace("tiny-aya-", "")


def main():
    print("Loading data...")
    mkqa_base, _, _ = load_all(DATA_ROOT, skip=SKIP_DIRS)
    mkqa_base["model"] = mkqa_base["model"].map(short_model)

    scs_df = mkqa_base.dropna(subset=["response_text"]).copy()
    reps = sorted(scs_df["rep"].unique())
    print(f"Found {len(reps)} reps: {reps}")

    print(f"Loading embedding model: {EMBED_MODEL}")
    encoder = SentenceTransformer(EMBED_MODEL)

    # Pre-compute all embeddings in one batch for speed
    print(f"Encoding {len(scs_df)} responses...")
    all_texts = scs_df["response_text"].tolist()
    all_embeddings = encoder.encode(all_texts, batch_size=BATCH_SIZE, show_progress_bar=True)
    scs_df["_emb_idx"] = range(len(scs_df))

    # Group by (sample_id, model, rep) and compute pairwise cosine similarity
    print("Computing SCS per (sample_id, model, rep)...")
    rows = []
    pair_rows = []
    groups = scs_df.groupby(["sample_id", "model", "rep"])
    for (sid, model, rep), grp in tqdm(groups, desc="SCS"):
        grp = grp.dropna(subset=["response_text"])
        if len(grp) < 2:
            continue
        langs = grp["language"].values
        idxs = grp["_emb_idx"].values
        embs = all_embeddings[idxs]
        sim = cosine_similarity(embs)
        triu = sim[np.triu_indices_from(sim, k=1)]
        rows.append({
            "sample_id": sid,
            "model": model,
            "rep": rep,
            "n_languages": len(grp),
            "scs": float(triu.mean()),
        })
        # Per-language-pair similarities
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                l1, l2 = sorted([langs[i], langs[j]])
                pair_rows.append({
                    "sample_id": sid,
                    "model": model,
                    "rep": rep,
                    "lang1": l1,
                    "lang2": l2,
                    "similarity": float(sim[i, j]),
                })

    results = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "scs_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved {len(results)} rows to {out_path}")

    pair_results = pd.DataFrame(pair_rows)
    pair_path = RESULTS_DIR / "scs_pairs.csv"
    pair_results.to_csv(pair_path, index=False)
    print(f"Saved {len(pair_results)} pair rows to {pair_path}")

    # Print summary (mean ± std across reps)
    rep_means = results.groupby(["model", "rep"])["scs"].mean().reset_index()
    summary = rep_means.groupby("model")["scs"].agg(["mean", "std"]).round(4)
    print("\nMean SCS by model (across reps):")
    print(summary.to_string())

    print("\nMean SCS by language pair:")
    pair_summary = pair_results.groupby(["lang1", "lang2"])["similarity"].mean().round(4)
    print(pair_summary.to_string())

    print("\nOverall mean SCS:", round(results["scs"].mean(), 4))

    # ── Within-language similarity (same prompt, same language, across reps) ──
    print("\nComputing within-language similarity across reps...")
    within_rows = []
    groups_within = scs_df.groupby(["sample_id", "model", "language"])
    for (sid, model, lang), grp in tqdm(groups_within, desc="Within-lang SCS"):
        grp = grp.dropna(subset=["response_text"])
        if len(grp) < 2:
            continue
        idxs = grp["_emb_idx"].values
        embs = all_embeddings[idxs]
        sim = cosine_similarity(embs)
        triu = sim[np.triu_indices_from(sim, k=1)]
        within_rows.append({
            "sample_id": sid,
            "model": model,
            "language": lang,
            "mean_sim": float(triu.mean()),
        })

    within_results = pd.DataFrame(within_rows)
    within_path = RESULTS_DIR / "scs_within_lang.csv"
    within_results.to_csv(within_path, index=False)
    print(f"Saved {len(within_results)} rows to {within_path}")

    within_summary = within_results.groupby("model")["mean_sim"].agg(["mean", "std"]).round(4)
    print("\nMean within-language similarity by model:")
    print(within_summary.to_string())


if __name__ == "__main__":
    main()
