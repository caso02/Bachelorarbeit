"""Cosine-similarity based nearest-neighbour search."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def find_top_k(
    embeddings: np.ndarray,
    query_index: int,
    top_k: int = 5,
    df: pd.DataFrame | None = None,
    text_col: str = "text_clean",
    id_col: str = "id",
) -> pd.DataFrame:
    """Return the top-k most similar posts for a given query index.

    Args:
        embeddings: Array of shape (N, D).
        query_index: Row index of the query post.
        top_k: Number of neighbours to return (excluding the query itself).
        df: Optional DataFrame to enrich results with text/ID columns.
        text_col: Name of the cleaned-text column.
        id_col: Name of the ID column.

    Returns:
        DataFrame with columns [rank, id, similarity, text].
    """
    if query_index < 0 or query_index >= len(embeddings):
        raise IndexError(
            f"query_index={query_index} ausserhalb des Bereichs [0, {len(embeddings) - 1}]"
        )

    query_vec = embeddings[query_index].reshape(1, -1)
    sims = cosine_similarity(query_vec, embeddings).flatten()
    sims[query_index] = -1.0  # exclude self

    top_indices = np.argsort(sims)[::-1][:top_k]

    rows = []
    for rank, idx in enumerate(top_indices, start=1):
        row: dict = {"rank": rank, "index": int(idx), "similarity": round(float(sims[idx]), 4)}
        if df is not None:
            row[id_col] = df.iloc[idx][id_col]
            row[text_col] = df.iloc[idx][text_col]
        rows.append(row)

    return pd.DataFrame(rows)
