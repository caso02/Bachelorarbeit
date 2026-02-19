"""Evaluation metrics and per-cluster analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

SOCIAL_MEDIA_STOPWORDS: list[str] = [
    "just", "new", "best", "year", "day", "got", "started", "really",
    "anyone", "update", "take", "today", "week", "weeks", "month", "months",
    "going", "love", "great", "good", "nice", "make", "made", "thing", "things",
]


def _build_stopwords() -> list[str]:
    """Combine sklearn's English stopwords with social-media filler words."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return list(set(ENGLISH_STOP_WORDS) | set(SOCIAL_MEDIA_STOPWORDS))


def compute_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Compute clustering quality metrics.

    Silhouette and Davies-Bouldin require at least 2 non-noise clusters.
    Returns a dict with the metric values (or None if not computable).
    """
    unique_labels = set(labels)
    unique_labels.discard(-1)

    if len(unique_labels) < 2:
        return {"silhouette": None, "davies_bouldin": None, "n_clusters": len(unique_labels)}

    mask = labels != -1
    if mask.sum() < 2:
        return {"silhouette": None, "davies_bouldin": None, "n_clusters": len(unique_labels)}

    sil = float(silhouette_score(embeddings[mask], labels[mask]))
    db = float(davies_bouldin_score(embeddings[mask], labels[mask]))

    return {
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "n_clusters": len(unique_labels),
        "noise_points": int((labels == -1).sum()),
    }


def extract_cluster_keywords(
    df: pd.DataFrame,
    labels: np.ndarray,
    text_col: str = "text_clean",
    top_n: int = 5,
) -> dict[int, list[str]]:
    """Extract top TF-IDF keywords (uni- and bigrams) per cluster.

    Uses combined English + social-media stopwords, ngram_range=(1,2),
    and adaptive min_df (falls back to 1 for very small clusters).
    """
    stop_words = _build_stopwords()
    result: dict[int, list[str]] = {}
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        mask = labels == label
        texts = df.loc[mask, text_col].tolist()
        if not texts:
            continue

        n_docs = len(texts)
        effective_min_df = 1 if n_docs <= 3 else 2
        effective_max_df = 1.0 if n_docs <= 5 else 0.7

        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=effective_min_df,
                max_df=effective_max_df,
                max_features=1000,
                stop_words=stop_words,
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            result[label] = []
            continue

        mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = mean_tfidf.argsort()[::-1][:top_n]
        feature_names = vectorizer.get_feature_names_out()
        result[label] = [feature_names[i] for i in top_indices]

    return result


def get_representative_examples(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    text_col: str = "text_clean",
    id_col: str = "id",
    n_examples: int = 3,
) -> dict[int, list[dict[str, Any]]]:
    """Pick representative examples per cluster (closest to centroid).

    Returns mapping from cluster_id -> list of dicts with id + text.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    result: dict[int, list[dict[str, Any]]] = {}
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if len(idx) == 0:
            continue

        cluster_emb = embeddings[idx]
        centroid = cluster_emb.mean(axis=0, keepdims=True)
        sims = cosine_similarity(centroid, cluster_emb).flatten()
        top = sims.argsort()[::-1][: min(n_examples, len(idx))]

        examples = []
        for t in top:
            global_idx = idx[t]
            examples.append({
                id_col: df.iloc[global_idx][id_col],
                text_col: df.iloc[global_idx][text_col],
            })
        result[label] = examples

    return result


# ---------------------------------------------------------------------------
# C) Weak-label evaluation (NMI, ARI, Purity)
# ---------------------------------------------------------------------------

def _purity(labels: np.ndarray, categories: np.ndarray, include_noise: bool) -> float | None:
    """Compute cluster purity against ground-truth categories.

    Args:
        labels: Predicted cluster labels (may contain -1 for noise).
        categories: Ground-truth category strings/ints, same length as labels.
        include_noise: If True, noise points (-1) are treated as their own
            pseudo-cluster. If False, noise points are excluded.

    Returns:
        Purity score in [0, 1], or None if no points remain after filtering.
    """
    if not include_noise:
        mask = labels != -1
        labels = labels[mask]
        categories = categories[mask]

    if len(labels) == 0:
        return None

    total = 0
    for cluster_id in np.unique(labels):
        cluster_cats = categories[labels == cluster_id]
        _, counts = np.unique(cluster_cats, return_counts=True)
        total += counts.max()

    return round(total / len(labels), 4)


def compute_weak_label_metrics(
    labels: np.ndarray,
    categories: np.ndarray,
) -> dict[str, Any]:
    """Compute NMI, ARI, and Purity of clustering vs. weak category labels.

    Noise handling:
        - NMI/ARI: computed on all points (noise = label -1 is a valid group).
        - purity: noise excluded (only real clusters).
        - purity_including_noise: noise treated as its own cluster.

    Returns dict with nmi, ari, purity, purity_including_noise.
    """
    mask_valid = labels != -1
    n_non_noise = int(mask_valid.sum())

    nmi = float(normalized_mutual_info_score(categories, labels))
    ari = float(adjusted_rand_score(categories, labels))
    pur = _purity(labels, categories, include_noise=False)
    pur_noise = _purity(labels, categories, include_noise=True)

    return {
        "nmi": round(nmi, 4),
        "ari": round(ari, 4),
        "purity": pur,
        "purity_including_noise": pur_noise,
        "n_evaluated": n_non_noise,
    }
