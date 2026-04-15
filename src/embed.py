"""Embedding generation with caching — supports SBERT and Gemini backends.

Backend selection is automatic based on the model name:
  - "gemini-*" or "models/gemini-*"  →  Google Gemini Embedding API
  - everything else                  →  Sentence-Transformers (SBERT)
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _is_gemini_model(model_name: str) -> bool:
    return model_name.startswith("gemini-") or model_name.startswith("models/gemini-")


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

_GEMINI_BATCH_SIZE = 100  # Gemini API supports up to 100 texts per batch call


def _embed_with_gemini(
    texts: list[str],
    model_name: str,
    output_dimensionality: Optional[int] = None,
    task_type: str = "CLUSTERING",
    api_key: Optional[str] = None,
) -> np.ndarray:
    """Call the Gemini Embedding API for a list of texts using batch requests.

    Args:
        texts: List of strings to embed.
        model_name: Full or short Gemini model name
            (e.g. "gemini-embedding-2-preview" or "models/gemini-embedding-2-preview").
        output_dimensionality: Reduce output to this many dimensions.
            None = model default (3072 for gemini-embedding-2-preview).
        task_type: Gemini task type hint.  Use "CLUSTERING" for clustering
            pipelines, "SEMANTIC_SIMILARITY" for similarity/readiness scores.
        api_key: Google API key.  Falls back to GOOGLE_API_KEY env var.

    Returns:
        Float64 array of shape (N, D).
    """
    from google.genai import types as genai_types
    from src.utils import get_gemini_client

    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Kein Google API Key gefunden. "
            "Setze GOOGLE_API_KEY als Umgebungsvariable oder trage ihn in model.gemini.api_key ein."
        )

    client = get_gemini_client(api_key=key)

    embed_config = genai_types.EmbedContentConfig(task_type=task_type)
    if output_dimensionality is not None:
        embed_config = genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=output_dimensionality,
        )

    all_embeddings: list[list[float]] = []

    # Process in batches for efficiency
    for batch_start in range(0, len(texts), _GEMINI_BATCH_SIZE):
        batch = texts[batch_start:batch_start + _GEMINI_BATCH_SIZE]

        # Retry loop for 429 rate-limit errors
        while True:
            try:
                result = client.models.embed_content(
                    model=model_name,
                    contents=batch,
                    config=embed_config,
                )
                for emb in result.embeddings:
                    all_embeddings.append(emb.values)
                break
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    match = re.search(r"retry in (\d+)", msg)
                    wait = int(match.group(1)) + 5 if match else 65
                    print(f"\n    ⏳ Rate limit (429) — warte {wait}s ... ", end="", flush=True)
                    time.sleep(wait)
                    print("weiter.")
                else:
                    raise

        done = min(batch_start + len(batch), len(texts))
        print(f"    {done}/{len(texts)} eingebettet...", end="\r")

    print()  # newline after progress
    return np.array(all_embeddings, dtype=np.float64)


# ---------------------------------------------------------------------------
# SBERT backend
# ---------------------------------------------------------------------------

def _embed_with_sbert(
    texts: list[str],
    model_name: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed texts using a Sentence-Transformers model."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return np.array(embeddings, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_or_compute_embeddings(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    id_col: str = "id",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    embeddings_path: Optional[str] = None,
    ids_path: Optional[str] = None,
    # Gemini-specific options (ignored for SBERT models)
    gemini_api_key: Optional[str] = None,
    gemini_output_dimensionality: Optional[int] = None,
    gemini_task_type: str = "CLUSTERING",
) -> tuple[np.ndarray, list[int | str]]:
    """Compute embeddings or load from cache if files exist and IDs match.

    Automatically selects the backend based on *model_name*:
    - ``gemini-*``  →  Gemini Embedding API
    - everything else  →  Sentence-Transformers

    Args:
        df: DataFrame with text and ID columns.
        text_col: Column containing preprocessed text.
        id_col: Column containing unique post IDs.
        model_name: Model identifier (SBERT path or Gemini model name).
        batch_size: Batch size for SBERT encoding (ignored for Gemini).
        embeddings_path: Path to .npy cache file.
        ids_path: Path to JSON file storing cached IDs.
        gemini_api_key: Google API key; falls back to GOOGLE_API_KEY env var.
        gemini_output_dimensionality: Optional dimension reduction (Gemini only).
        gemini_task_type: Gemini task type hint (default "CLUSTERING").

    Returns:
        Tuple of (embeddings array [N, D], list of post IDs).
    """
    ids = df[id_col].tolist()

    # --- Cache check ---
    if embeddings_path and ids_path:
        emb_file = Path(embeddings_path)
        id_file = Path(ids_path)
        if emb_file.exists() and id_file.exists():
            cached_ids = json.loads(id_file.read_text())
            if cached_ids == ids:
                print(f"  ↳ Embeddings aus Cache geladen ({emb_file})")
                return np.load(emb_file), ids
            print("  ↳ Cache-IDs stimmen nicht überein – berechne neu")

    texts = df[text_col].tolist()
    if not texts:
        raise ValueError("Keine Texte zum Einbetten vorhanden")

    # --- Compute ---
    if _is_gemini_model(model_name):
        dim_info = gemini_output_dimensionality or "default (3072)"
        print(f"  ↳ Gemini Embedding: {model_name}")
        print(f"    task_type={gemini_task_type}, output_dimensionality={dim_info}")
        embeddings = _embed_with_gemini(
            texts,
            model_name=model_name,
            output_dimensionality=gemini_output_dimensionality,
            task_type=gemini_task_type,
            api_key=gemini_api_key,
        )
    else:
        print(f"  ↳ Lade SBERT-Modell: {model_name}")
        print(f"  ↳ Berechne Embeddings für {len(texts)} Texte (batch_size={batch_size})")
        embeddings = _embed_with_sbert(texts, model_name=model_name, batch_size=batch_size)

    print(f"  ↳ Embedding-Shape: {embeddings.shape}")

    # --- Persist cache ---
    if embeddings_path and ids_path:
        emb_file = Path(embeddings_path)
        id_file = Path(ids_path)
        emb_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_file, embeddings)
        id_file.write_text(json.dumps(ids))
        print(f"  ↳ Embeddings gespeichert → {emb_file}")

    return embeddings, ids


def calculate_readiness_score(
    embeddings: np.ndarray,
    anchor_text: str = "Ich möchte mich mit Gleichgesinnten vernetzen",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    gemini_api_key: Optional[str] = None,
    gemini_output_dimensionality: Optional[int] = None,
    gemini_task_type: str = "SEMANTIC_SIMILARITY",
) -> np.ndarray:
    """Compute community-readiness scores via cosine similarity to an anchor.

    Embeds *anchor_text* with the same model/backend as the posts, then
    returns the cosine similarity between every post embedding and the anchor.

    Typical use: measure how strongly each post signals intent to connect
    with like-minded people (Action-Agent feature).

    Args:
        embeddings: Post embedding matrix of shape (N, D).
        anchor_text: Reference sentence representing the target intent.
        model_name: Model used to embed the posts (determines backend).
        gemini_api_key: Google API key (Gemini only).
        gemini_output_dimensionality: Dimension reduction (Gemini only).
            Must match the dimensionality used for *embeddings*.
        gemini_task_type: Gemini task type for the anchor embedding.
            "SEMANTIC_SIMILARITY" is recommended for intent matching.

    Returns:
        1-D float64 array of shape (N,) with similarity scores in [-1, 1].
    """
    if _is_gemini_model(model_name):
        anchor_emb = _embed_with_gemini(
            [anchor_text],
            model_name=model_name,
            output_dimensionality=gemini_output_dimensionality,
            task_type=gemini_task_type,
            api_key=gemini_api_key,
        )
    else:
        anchor_emb = _embed_with_sbert([anchor_text], model_name=model_name)

    scores = cosine_similarity(embeddings, anchor_emb).flatten()
    return scores.astype(np.float64)
