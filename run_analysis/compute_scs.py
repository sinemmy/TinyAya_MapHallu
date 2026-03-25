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

    scs_df = mkqa_base[mkqa_base["rep"] == 0].copy()
    scs_df = scs_df.dropna(subset=["response_text"])

    print(f"Loading embedding model: {EMBED_MODEL}")
    encoder = SentenceTransformer(EMBED_MODEL)

    # Pre-compute all embeddings in one batch for speed
    print(f"Encoding {len(scs_df)} responses...")
    all_texts = scs_df["response_text"].tolist()
    all_embeddings = encoder.encode(all_texts, batch_size=BATCH_SIZE, show_progress_bar=True)
    scs_df["_emb_idx"] = range(len(scs_df))

    # Group by (sample_id, model) and compute pairwise cosine similarity
    print("Computing SCS per (sample_id, model)...")
    rows = []
    groups = scs_df.groupby(["sample_id", "model"])
    for (sid, model), grp in tqdm(groups, desc="SCS"):
        grp = grp.dropna(subset=["response_text"])
        if len(grp) < 2:
            continue
        idxs = grp["_emb_idx"].values
        embs = all_embeddings[idxs]
        sim = cosine_similarity(embs)
        triu = sim[np.triu_indices_from(sim, k=1)]
        rows.append({
            "sample_id": sid,
            "model": model,
            "n_languages": len(grp),
            "scs": float(triu.mean()),
        })

    results = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "scs_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved {len(results)} rows to {out_path}")

    # Print summary
    print("\nMean SCS by model:")
    print(results.groupby("model")["scs"].mean().round(4).to_string())

    print("\nOverall mean SCS:", round(results["scs"].mean(), 4))


if __name__ == "__main__":
    main()
