"""Clustering algorithms for post embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans

from src.evaluate import compute_metrics


def run_clustering(
    embeddings: np.ndarray,
    method: str,
    params: dict[str, Any],
) -> np.ndarray:
    """Run a clustering algorithm and return label array.

    Args:
        embeddings: Array of shape (N, D).
        method: One of 'kmeans', 'dbscan', 'hdbscan'.
        params: Algorithm-specific keyword arguments.

    Returns:
        Array of cluster labels (length N). -1 means noise (DBSCAN/HDBSCAN).
    """
    method = method.lower()

    if method == "kmeans":
        model = KMeans(**params)
        return model.fit_predict(embeddings)

    if method == "dbscan":
        model = DBSCAN(**params)
        return model.fit_predict(embeddings)

    if method == "hdbscan":
        try:
            import hdbscan as hdb
        except ImportError:
            print("  ⚠ hdbscan nicht installiert – überspringe HDBSCAN")
            return np.full(len(embeddings), -1, dtype=int)
        model = hdb.HDBSCAN(**params)
        return model.fit_predict(embeddings)

    raise ValueError(f"Unbekannte Clustering-Methode: {method}")


def run_dbscan_grid_search(
    embeddings: np.ndarray,
    eps_values: list[float],
    min_samples_values: list[int],
    metric: str = "cosine",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run DBSCAN over a grid of (eps, min_samples) and select the best combo.

    Selection strategy:
        1. Highest silhouette score (requires >= 2 clusters, not all noise).
        2. Tie-breaker: fewer noise points.
        3. Tie-breaker: moderate cluster count (avoid extremes).

    Args:
        embeddings: Array of shape (N, D).
        eps_values: List of eps values to try.
        min_samples_values: List of min_samples values to try.
        metric: Distance metric passed to DBSCAN (default: cosine).

    Returns:
        Tuple of (results DataFrame, best_info dict with keys
        eps, min_samples, labels, metrics).
    """
    rows: list[dict[str, Any]] = []

    for eps in eps_values:
        for ms in min_samples_values:
            labels = DBSCAN(eps=eps, min_samples=ms, metric=metric).fit_predict(embeddings)
            metrics = compute_metrics(embeddings, labels)
            rows.append({
                "eps": eps,
                "min_samples": ms,
                "n_clusters": metrics["n_clusters"],
                "n_noise": int((labels == -1).sum()),
                "silhouette": metrics.get("silhouette"),
                "davies_bouldin": metrics.get("davies_bouldin"),
            })

    grid_df = pd.DataFrame(rows)

    valid = grid_df[grid_df["silhouette"].notna()].copy()
    if valid.empty:
        best_row = grid_df.iloc[0]
    else:
        valid = valid.sort_values(
            by=["silhouette", "n_noise", "n_clusters"],
            ascending=[False, True, True],
        )
        best_row = valid.iloc[0]

    best_eps = best_row["eps"]
    best_ms = int(best_row["min_samples"])
    best_labels = DBSCAN(
        eps=best_eps, min_samples=best_ms, metric=metric,
    ).fit_predict(embeddings)
    best_metrics = compute_metrics(embeddings, best_labels)

    best_info: dict[str, Any] = {
        "eps": best_eps,
        "min_samples": best_ms,
        "labels": best_labels,
        "metrics": best_metrics,
    }

    return grid_df, best_info
