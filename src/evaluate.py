"""Evaluation metrics, per-cluster analysis, and backend comparison."""

from __future__ import annotations

import json
from pathlib import Path
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


# ---------------------------------------------------------------------------
# Flat clustering metrics (unchanged)
# ---------------------------------------------------------------------------

def compute_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Compute clustering quality metrics.

    Silhouette and Davies-Bouldin require at least 2 non-noise clusters.
    """
    unique_labels = set(labels)
    unique_labels.discard(-1)

    if len(unique_labels) < 2:
        return {"silhouette": None, "davies_bouldin": None, "n_clusters": len(unique_labels)}

    mask = labels != -1
    n_valid = int(mask.sum())
    if n_valid < len(unique_labels) + 1:
        return {"silhouette": None, "davies_bouldin": None, "n_clusters": len(unique_labels),
                "noise_points": int((labels == -1).sum())}

    try:
        sil = float(silhouette_score(embeddings[mask], labels[mask]))
        db = float(davies_bouldin_score(embeddings[mask], labels[mask]))
    except ValueError:
        return {"silhouette": None, "davies_bouldin": None, "n_clusters": len(unique_labels),
                "noise_points": int((labels == -1).sum())}

    return {
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "n_clusters": len(unique_labels),
        "noise_points": int((labels == -1).sum()),
    }


# ---------------------------------------------------------------------------
# TF-IDF keyword extraction
# ---------------------------------------------------------------------------

def extract_cluster_keywords(
    df: pd.DataFrame,
    labels: np.ndarray,
    text_col: str = "text_clean",
    top_n: int = 5,
) -> dict[int, list[str]]:
    """Extract top TF-IDF keywords (uni- and bigrams) per cluster."""
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


def extract_keywords_from_texts(
    texts: list[str],
    top_n: int = 5,
) -> list[str]:
    """Extract top TF-IDF keywords from a flat list of texts."""
    stop_words = _build_stopwords()
    if not texts:
        return []

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
        return []

    mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = mean_tfidf.argsort()[::-1][:top_n]
    feature_names = vectorizer.get_feature_names_out()
    return [feature_names[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Representative examples
# ---------------------------------------------------------------------------

def get_representative_examples(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    text_col: str = "text_clean",
    id_col: str = "id",
    n_examples: int = 3,
) -> dict[int, list[dict[str, Any]]]:
    """Pick representative examples per cluster (closest to centroid)."""
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


def get_representative_examples_from_indices(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    indices: np.ndarray,
    text_col: str = "text_clean",
    id_col: str = "id",
    n_examples: int = 3,
) -> list[dict[str, Any]]:
    """Pick examples closest to centroid from a subset of global indices."""
    from sklearn.metrics.pairwise import cosine_similarity

    if len(indices) == 0:
        return []

    sub_emb = embeddings[indices]
    centroid = sub_emb.mean(axis=0, keepdims=True)
    sims = cosine_similarity(centroid, sub_emb).flatten()
    top = sims.argsort()[::-1][: min(n_examples, len(indices))]

    examples = []
    for t in top:
        gi = indices[t]
        examples.append({
            id_col: df.iloc[gi][id_col],
            text_col: df.iloc[gi][text_col],
        })
    return examples


# ---------------------------------------------------------------------------
# Weak-label evaluation (NMI, ARI, Purity)
# ---------------------------------------------------------------------------

def _purity(labels: np.ndarray, categories: np.ndarray, include_noise: bool) -> float | None:
    """Compute cluster purity against ground-truth categories."""
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
    """Compute NMI, ARI, and Purity of clustering vs. weak category labels."""
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


# ---------------------------------------------------------------------------
# Stage 1: Routing accuracy
# ---------------------------------------------------------------------------

def compute_routing_accuracy(
    routing_df: pd.DataFrame,
    categories: np.ndarray,
) -> dict[str, Any]:
    """Compute Accuracy@1 and Accuracy@2 for topic routing vs. ground truth.

    Args:
        routing_df: DataFrame with columns topic_1, topic_2, ... as produced
            by ``routing.route_posts()``.
        categories: Ground-truth category array of same length.

    Returns:
        Dict with accuracy_at_1, accuracy_at_2, per_category breakdown.
    """
    n = len(categories)
    correct_at_1 = 0
    correct_at_2 = 0

    for i in range(n):
        gt = str(categories[i]).lower()
        t1 = str(routing_df.iloc[i].get("topic_1", "")).lower()
        t2 = str(routing_df.iloc[i].get("topic_2", "")).lower()

        if t1 == gt:
            correct_at_1 += 1
            correct_at_2 += 1
        elif t2 == gt:
            correct_at_2 += 1

    per_cat: dict[str, dict[str, float]] = {}
    for cat in sorted(set(categories)):
        mask = categories == cat
        cat_n = int(mask.sum())
        cat_correct = sum(
            1 for i in range(n)
            if mask[i] and str(routing_df.iloc[i].get("topic_1", "")).lower() == str(cat).lower()
        )
        per_cat[str(cat)] = {
            "n": cat_n,
            "accuracy_at_1": round(cat_correct / cat_n, 4) if cat_n else 0,
        }

    return {
        "accuracy_at_1": round(correct_at_1 / n, 4),
        "accuracy_at_2": round(correct_at_2 / n, 4),
        "n_posts": n,
        "per_category": per_cat,
    }


# ---------------------------------------------------------------------------
# Stage 2: Per-topic clustering metrics
# ---------------------------------------------------------------------------

def compute_per_topic_metrics(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    hierarchy_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Compute Silhouette / DBI per topic based on subcluster assignments.

    Args:
        df: Preprocessed DataFrame (must be index-aligned with embeddings).
        embeddings: Full embedding matrix (N, D).
        hierarchy_df: Output of ``hierarchical.cluster_per_topic()``.

    Returns:
        Dict ``{topic: {silhouette, davies_bouldin, n_subclusters, ...}}``.
    """
    id_to_idx = {row_id: i for i, row_id in enumerate(df["id"])}
    result: dict[str, dict[str, Any]] = {}

    for topic in sorted(hierarchy_df["topic_label"].unique()):
        topic_rows = hierarchy_df[hierarchy_df["topic_label"] == topic]
        global_indices = np.array([id_to_idx[rid] for rid in topic_rows["id"] if rid in id_to_idx])
        sub_labels = topic_rows["subcluster_id"].values

        if len(global_indices) == 0:
            continue

        topic_emb = embeddings[global_indices]
        metrics = compute_metrics(topic_emb, sub_labels)
        metrics["n_posts"] = len(global_indices)
        metrics["method_used"] = topic_rows["method_used"].iloc[0]
        result[topic] = metrics

    return result


# ---------------------------------------------------------------------------
# MRL truncation (Matryoshka Representation Learning)
# ---------------------------------------------------------------------------

def truncate_mrl(embeddings: np.ndarray, target_dim: int) -> np.ndarray:
    """Truncate embeddings to *target_dim* and L2-normalise (MRL reduction).

    Gemini Embedding 2 is trained with Matryoshka Representation Learning,
    meaning the first N dimensions of any full embedding already form a
    valid, useful embedding at reduced size.  This avoids extra API calls
    when evaluating multiple dimensionalities.

    Args:
        embeddings: Full embedding matrix of shape (N, D) where D >= target_dim.
        target_dim: Number of dimensions to keep.

    Returns:
        L2-normalised array of shape (N, target_dim).
    """
    if target_dim >= embeddings.shape[1]:
        return embeddings.copy()
    truncated = embeddings[:, :target_dim].copy()
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    return truncated / np.clip(norms, 1e-12, None)


# ---------------------------------------------------------------------------
# Backend comparison table
# ---------------------------------------------------------------------------

def generate_comparison_table(
    backends: list[dict[str, Any]],
    df: pd.DataFrame,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Generate a Markdown + CSV comparison table across embedding backends.

    Each entry in *backends* must be a dict with keys:
        - ``name``              (str)  display name, e.g. "Gemini 3072"
        - ``embeddings``        (np.ndarray)  shape (N, D)
        - ``routing_df``        (pd.DataFrame)  output of route_posts()
        - ``hierarchy_df``      (pd.DataFrame)  output of cluster_per_topic()
        - ``categories``        (np.ndarray | None)  ground-truth labels
        - ``readiness_scores``  (np.ndarray)  shape (N,)
        - ``stability_summary`` (dict | None)  loaded from stability_summary.json

    The function writes:
        - ``<output_dir>/comparison_table.csv``
        - ``<output_dir>/comparison_table.md``

    Returns:
        Summary DataFrame (one row per backend).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for b in backends:
        name = b["name"]
        emb = b["embeddings"]
        routing_df = b["routing_df"]
        hierarchy_df = b["hierarchy_df"]
        cats = b.get("categories")
        readiness = b.get("readiness_scores")
        stab = b.get("stability_summary")

        row: dict[str, Any] = {"Backend": name, "Dim": emb.shape[1]}

        # --- Routing accuracy ---
        if cats is not None:
            acc = compute_routing_accuracy(routing_df, cats)
            row["Acc@1"] = acc["accuracy_at_1"]
            row["Acc@2"] = acc["accuracy_at_2"]
        else:
            row["Acc@1"] = row["Acc@2"] = None

        # --- Mean Silhouette (Stage 2, across topics) ---
        topic_metrics = compute_per_topic_metrics(df, emb, hierarchy_df)
        sil_vals = [v["silhouette"] for v in topic_metrics.values() if v.get("silhouette") is not None]
        row["Sil (Ø)"] = round(float(np.mean(sil_vals)), 4) if sil_vals else None

        # --- Weak-label NMI / ARI (Stage 1 topic vs. category) ---
        if cats is not None:
            topic_labels_raw = routing_df["topic_1"].values
            unique_topics = sorted(set(topic_labels_raw))
            topic_to_int = {t: i for i, t in enumerate(unique_topics)}
            topic_ints = np.array([topic_to_int[t] for t in topic_labels_raw])
            wl = compute_weak_label_metrics(topic_ints, cats)
            row["NMI"] = wl["nmi"]
            row["ARI (Label)"] = wl["ari"]
        else:
            row["NMI"] = row["ARI (Label)"] = None

        # --- Stability ARI (from stability_summary.json) ---
        if stab and "stage2_subcluster_ari" in stab:
            ari_vals = [
                v["mean_ari"]
                for v in stab["stage2_subcluster_ari"].values()
                if isinstance(v, dict) and v.get("mean_ari") is not None
            ]
            std_vals = [
                v["std_ari"]
                for v in stab["stage2_subcluster_ari"].values()
                if isinstance(v, dict) and v.get("std_ari") is not None
            ]
            if ari_vals:
                row["ARI (Stab)"] = f"{np.mean(ari_vals):.3f}±{np.mean(std_vals):.3f}"
            else:
                row["ARI (Stab)"] = "—"
        else:
            row["ARI (Stab)"] = "—"

        # --- Community readiness ---
        if readiness is not None and len(readiness) > 0:
            row["Readiness (Ø)"] = round(float(np.mean(readiness)), 4)
        else:
            row["Readiness (Ø)"] = None

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # --- Save CSV ---
    csv_path = output_dir / "comparison_table.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  ↳ Comparison CSV → {csv_path}")

    # --- Save Markdown ---
    md_path = output_dir / "comparison_table.md"
    _write_comparison_markdown(summary_df, md_path)
    print(f"  ↳ Comparison Markdown → {md_path}")

    return summary_df


def _write_comparison_markdown(df: pd.DataFrame, path: Path) -> None:
    """Write a styled Markdown comparison table."""
    lines = [
        "# Backend-Vergleich: SBERT vs. Gemini",
        "",
        "_Generiert von `src/evaluate.py:generate_comparison_table()`_",
        "",
    ]

    # Header
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    # Rows
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if v is None:
                vals.append("—")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Legende",
        "- **Dim**: Embedding-Dimensionalität",
        "- **Acc@1 / Acc@2**: Routing-Accuracy (Stage 1)",
        "- **Sil (Ø)**: Mittlerer Silhouette-Score über alle Topics (Stage 2)",
        "- **NMI / ARI (Label)**: Übereinstimmung mit Weak-Labels (Stage 1 Topics vs. Kategorien)",
        "- **ARI (Stab)**: Mittlerer paarweiser ARI aus Stabilitäts-Runs (Stage 2)",
        "- **Readiness (Ø)**: Mittlere Cosine-Similarity zum Community-Anker",
        "- **—**: Wert nicht verfügbar (kein Stabilitäts-Run durchgeführt)",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
