"""Stage 2: Per-topic sub-clustering to form a Topic → Subcluster hierarchy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def cluster_per_topic(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    topic_labels: np.ndarray,
    method: str = "hdbscan",
    method_params: dict[str, Any] | None = None,
    kmeans_fallback_k: int = 3,
    min_samples_for_hdbscan: int = 6,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run clustering independently within each routed topic.

    Falls back to KMeans when the topic group is too small for HDBSCAN
    or when HDBSCAN is unavailable.

    Args:
        df: DataFrame with at least 'id' and 'text_clean'.
        embeddings: Full embedding matrix (N, D).
        topic_labels: Array of topic strings, one per row in df.
        method: Primary clustering method ('hdbscan' or 'kmeans').
        method_params: Extra kwargs for the clustering algorithm.
        kmeans_fallback_k: Number of clusters for KMeans fallback.
        min_samples_for_hdbscan: Minimum group size to attempt HDBSCAN.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with columns:
            id, topic_label, subcluster_id, method_used, n_in_topic
    """
    if method_params is None:
        method_params = {}

    results: list[dict[str, Any]] = []
    unique_topics = sorted(set(topic_labels))

    for topic in unique_topics:
        mask = topic_labels == topic
        idx = np.where(mask)[0]
        n = len(idx)
        topic_emb = embeddings[idx]

        if n < 2:
            sub_labels = np.zeros(n, dtype=int)
            used_method = "single"
        elif method == "hdbscan" and n >= min_samples_for_hdbscan:
            sub_labels, used_method = _try_hdbscan(topic_emb, method_params, random_state, kmeans_fallback_k)
        else:
            k = min(kmeans_fallback_k, n)
            sub_labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(topic_emb)
            used_method = "kmeans"

        for local_i, global_i in enumerate(idx):
            results.append({
                "id": df.iloc[global_i]["id"],
                "topic_label": topic,
                "subcluster_id": int(sub_labels[local_i]),
                "method_used": used_method,
                "n_in_topic": n,
            })

    return pd.DataFrame(results)


def _try_hdbscan(
    embeddings: np.ndarray,
    params: dict[str, Any],
    random_state: int,
    kmeans_fallback_k: int,
) -> tuple[np.ndarray, str]:
    """Attempt HDBSCAN; fall back to KMeans if import fails or all noise."""
    try:
        import hdbscan as hdb
    except ImportError:
        k = min(kmeans_fallback_k, len(embeddings))
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(embeddings)
        return labels, "kmeans"

    p = {**params}
    p.setdefault("min_cluster_size", max(3, len(embeddings) // 10))
    p.setdefault("min_samples", 2)

    model = hdb.HDBSCAN(**p)
    labels = model.fit_predict(embeddings)

    n_clusters = len(set(labels) - {-1})
    if n_clusters == 0:
        k = min(kmeans_fallback_k, len(embeddings))
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(embeddings)
        return labels, "kmeans_fallback"

    return labels, "hdbscan"
