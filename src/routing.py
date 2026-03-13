"""Stage 1: Topic routing via cosine similarity against label prototypes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def embed_label_prototypes(
    interest_labels: dict[str, list[str]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    cache_dir: str | None = None,
    gemini_cfg: Optional[dict[str, Any]] = None,
) -> dict[str, np.ndarray]:
    """Embed seed sentences for each interest label and return per-label centroids.

    Uses the same backend as the post embeddings (SBERT or Gemini) so that
    prototypes and posts live in the same embedding space.

    Args:
        interest_labels: Mapping ``{topic_name: [seed_sentence, ...]}``.
        model_name: Model identifier — determines the backend automatically
            (``gemini-*`` → Gemini API, everything else → Sentence-Transformers).
        batch_size: Encoding batch size (SBERT only; ignored for Gemini).
        cache_dir: If set, cache individual label embeddings as .npy files.
        gemini_cfg: Optional dict with Gemini-specific options:
            ``api_key``, ``output_dimensionality``, ``task_type``.
            Ignored when using a SBERT model.

    Returns:
        Dict mapping topic name to its prototype embedding (1-D centroid).
    """
    from src.embed import _is_gemini_model, _embed_with_gemini, _embed_with_sbert

    gemini_cfg = gemini_cfg or {}
    gemini_dim = gemini_cfg.get("output_dimensionality")
    gemini_task = gemini_cfg.get("task_type", "CLUSTERING")
    gemini_key = gemini_cfg.get("api_key")

    cache_path = Path(cache_dir) if cache_dir else None
    meta_path = cache_path / "proto_meta.json" if cache_path else None

    # --- Cache check ---
    if cache_path and meta_path and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        model_match = meta.get("model") == model_name
        labels_match = set(meta.get("labels", [])) == set(interest_labels)
        dim_match = meta.get("output_dimensionality") == gemini_dim
        if model_match and labels_match and dim_match:
            prototypes: dict[str, np.ndarray] = {}
            for label in interest_labels:
                npy = cache_path / f"proto_{label}.npy"
                if npy.exists():
                    prototypes[label] = np.load(npy)
            if len(prototypes) == len(interest_labels):
                print("  ↳ Label-Prototypen aus Cache geladen")
                return prototypes

    print(f"  ↳ Berechne Label-Prototypen mit {model_name}")

    prototypes = {}
    for label, sentences in interest_labels.items():
        if _is_gemini_model(model_name):
            embs = _embed_with_gemini(
                sentences,
                model_name=model_name,
                output_dimensionality=gemini_dim,
                task_type=gemini_task,
                api_key=gemini_key,
            )
        else:
            embs = _embed_with_sbert(sentences, model_name=model_name, batch_size=batch_size)

        prototypes[label] = embs.mean(axis=0)

    # --- Persist cache ---
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
        for label, vec in prototypes.items():
            np.save(cache_path / f"proto_{label}.npy", vec)
        meta_path.write_text(json.dumps({
            "model": model_name,
            "labels": list(interest_labels.keys()),
            "output_dimensionality": gemini_dim,
        }))

    return prototypes


def route_posts(
    post_embeddings: np.ndarray,
    prototypes: dict[str, np.ndarray],
    top_n: int = 2,
) -> pd.DataFrame:
    """Route each post to the most similar topic label(s).

    Args:
        post_embeddings: Array of shape (N, D).
        prototypes: Dict ``{topic_name: centroid_vector}``.
        top_n: Number of top topics to return per post.

    Returns:
        DataFrame with columns:
            - topic_1, score_1, topic_2, score_2, ... (up to top_n)
    """
    label_names = list(prototypes.keys())
    proto_matrix = np.stack([prototypes[l] for l in label_names])  # (K, D)

    sims = cosine_similarity(post_embeddings, proto_matrix)  # (N, K)

    rows: list[dict[str, Any]] = []
    for i in range(len(post_embeddings)):
        row: dict[str, Any] = {}
        ranked = np.argsort(sims[i])[::-1]
        for rank in range(min(top_n, len(label_names))):
            idx = ranked[rank]
            row[f"topic_{rank + 1}"] = label_names[idx]
            row[f"score_{rank + 1}"] = round(float(sims[i, idx]), 4)
        rows.append(row)

    return pd.DataFrame(rows)
