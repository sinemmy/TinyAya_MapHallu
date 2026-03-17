# Semantic Consistency Score (SCS) computation.

import numpy as np
from functools import lru_cache


# paraphrase-multilingual-mpnet-base-v2 supports 50+ languages including
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"


@lru_cache(maxsize=1)
def _load_model():
    """
    Load the embedding model once and cache it for the process lifetime.
    First call takes ~10 seconds to download/load. All subsequent calls
    return the cached model instantly.
    """
    from sentence_transformers import SentenceTransformer
    print(f"[embeddings] Loading {EMBEDDING_MODEL_NAME}...")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts. Returns array of shape (n, embedding_dim).
    Filters out None and empty strings before embedding.
    """
    model = _load_model()
    valid = [t for t in texts if t]
    if not valid:
        return np.array([])
    return model.encode(valid, convert_to_numpy=True, show_progress_bar=False)


def pairwise_cosine_similarity(embeddings: np.ndarray) -> float | None:
    """
    Computes mean pairwise cosine similarity across all pairs in the array.

    Interpretation:
      High SCS (~1.0) — model gives semantically similar answers regardless
                        of which language the prompt was in. Consistent.
      Low SCS  (~0.0) — model answers diverge depending on language.
                        Potential hallucination or knowledge gap in some languages.

    Returns None if fewer than 2 embeddings (can't compute pairs).
    """
    if len(embeddings) < 2:
        return None

    # L2 normalise so dot product equals cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms = np.where(norms == 0, 1, norms)
    normalised = embeddings / norms

    similarities = []
    n = len(normalised)
    for i in range(n):
        for j in range(i + 1, n):
            similarities.append(float(np.dot(normalised[i], normalised[j])))

    return round(float(np.mean(similarities)), 4)


def compute_scs(responses: list[str | None]) -> float | None:
    valid = [r for r in responses if r]
    if len(valid) < 2:
        return None
    embeddings = embed(valid)
    return pairwise_cosine_similarity(embeddings)