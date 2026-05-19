"""Discovery module — semantic search over pre-built corpus index.

Finds posts from the Evolution-AI corpus that are semantically similar
to a user query, then feeds them into the OODA pipeline for community
detection and nudge generation.

Usage:
    from src.discovery import discover
    state = discover("Immobilien-Investments", top_k=200, on_phase=callback)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def _load_corpus(
    corpus_path: Path,
) -> tuple[np.ndarray, list[str], pd.DataFrame, dict]:
    """Load pre-built corpus index.

    Returns:
        embeddings: np.ndarray shape (N, dim), mmap mode 'r'
        post_ids: list of N string IDs
        posts_df: DataFrame with id, text_clean, subreddit, category_1
        manifest: dict from manifest.json

    Raises:
        FileNotFoundError: if corpus index files are missing
        ValueError: if manifest model doesn't match expected model
    """
    corpus_path = Path(corpus_path)

    manifest_path = corpus_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Corpus-Index nicht gefunden: {manifest_path}\n"
            "Bitte zuerst ausführen: python scripts/build_corpus_index.py"
        )

    manifest = json.loads(manifest_path.read_text())

    emb_path = corpus_path / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings nicht gefunden: {emb_path}")

    embeddings = np.load(emb_path, mmap_mode="r")

    ids_path = corpus_path / "post_ids.json"
    post_ids = json.loads(ids_path.read_text())

    posts_path = corpus_path / "posts.parquet"
    posts_df = pd.read_parquet(posts_path)

    # Validate dimensions
    if embeddings.shape[0] != len(post_ids):
        raise ValueError(
            f"Embedding-Anzahl ({embeddings.shape[0]}) stimmt nicht mit "
            f"post_ids ({len(post_ids)}) überein"
        )

    return embeddings, post_ids, posts_df, manifest


# ---------------------------------------------------------------------------
# Query embedding
# ---------------------------------------------------------------------------

def _embed_query(
    query: str,
    model_name: str,
    dim: int = 768,
) -> np.ndarray:
    """Embed a single query string, truncate to dim via MRL.

    Uses task_type="CLUSTERING" to match corpus embedding space.

    Returns:
        np.ndarray shape (1, dim), float32
    """
    from src.embed import _embed_with_gemini
    from src.evaluate import truncate_mrl

    # Embed as single-item batch
    emb_full = _embed_with_gemini(
        [query],
        model_name=model_name,
        task_type="CLUSTERING",
    )
    # MRL truncation + L2 norm
    emb_truncated = truncate_mrl(emb_full, dim)
    return emb_truncated.astype(np.float32)


# ---------------------------------------------------------------------------
# Corpus search
# ---------------------------------------------------------------------------

def _search_corpus(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    top_k: int = 200,
    min_similarity: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force cosine similarity search.

    Args:
        query_embedding: shape (1, dim)
        corpus_embeddings: shape (N, dim)
        top_k: maximum number of results
        min_similarity: minimum cosine similarity threshold (0.0 = disabled)

    Returns:
        indices: np.ndarray of matched corpus indices
        similarities: np.ndarray of corresponding similarity scores
    """
    sims = cosine_similarity(query_embedding, corpus_embeddings).flatten()
    ranked = np.argsort(sims)[::-1][:top_k]
    scores = sims[ranked]

    if min_similarity > 0.0:
        mask = scores >= min_similarity
        ranked = ranked[mask]
        scores = scores[mask]

    return ranked, scores


# ---------------------------------------------------------------------------
# Post loading
# ---------------------------------------------------------------------------

def _load_matched_posts(
    indices: np.ndarray,
    similarities: np.ndarray,
    post_ids: list[str],
    posts_df: pd.DataFrame,
) -> list[dict]:
    """Build post dicts from matched indices.

    Returns list of dicts compatible with AgentState.posts format:
        {id, text_clean, category, author_hash, score, query_similarity}
    """
    matched_ids = [post_ids[i] for i in indices]
    matched_df = posts_df[posts_df["id"].isin(set(matched_ids))].copy()

    # Build lookup for similarity scores
    sim_lookup = {post_ids[idx]: float(sim) for idx, sim in zip(indices, similarities)}

    posts = []
    for _, row in matched_df.iterrows():
        posts.append({
            "id": str(row["id"]),
            "text_clean": row["text_clean"],
            "category": row.get("category_1", ""),
            "author_hash": "",
            "score": 0,
            "query_similarity": sim_lookup.get(str(row["id"]), 0.0),
        })

    # Sort by similarity (highest first)
    posts.sort(key=lambda p: p["query_similarity"], reverse=True)
    return posts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def discover(
    query: str,
    top_k: int = 200,
    min_similarity: float = 0.0,
    corpus_path: str | Path = "data/corpus_index/",
    on_phase: Optional[Callable] = None,
    output_dir: Optional[Path] = None,
) -> "AgentState":
    """Search corpus + run OODA pipeline.

    1. Load corpus index
    2. Embed query
    3. Cosine similarity search → top_k posts
    4. Create AgentState with matched posts
    5. Run Coordinator pipeline (skips Collect phase)
    6. Return final state with nudges + ethics reviews

    Args:
        query: Free-text search query (e.g. "Immobilien-Investments")
        top_k: Maximum number of posts to retrieve (default: 200)
        min_similarity: Minimum cosine similarity (0.0 = disabled)
        corpus_path: Path to corpus index directory
        on_phase: Optional callback for UI progress updates
        output_dir: Optional output directory for results

    Returns:
        AgentState with populated posts, clusters, nudges, final_nudges

    Raises:
        ValueError: if query is empty or model mismatch
        FileNotFoundError: if corpus index is missing
    """
    from src.agent_state import AgentState
    from src.community_crew import Coordinator

    # Validate query
    if not query or not query.strip():
        raise ValueError("Query darf nicht leer sein.")

    corpus_path = Path(corpus_path)
    if not corpus_path.is_absolute():
        corpus_path = PROJECT_ROOT / corpus_path

    # --- 1. Load corpus ---
    print(f"\n🔍 Discovery: '{query}' (top_k={top_k}, min_sim={min_similarity})")
    embeddings, post_ids, posts_df, manifest = _load_corpus(corpus_path)
    print(f"  Korpus: {len(post_ids)} Posts, {manifest['n_categories']} Kategorien, "
          f"{manifest['dim']}d Embeddings")

    # --- 2. Embed query ---
    print(f"  Query embedden ({manifest['model_name']}) ...")
    query_emb = _embed_query(query, manifest["model_name"], manifest["dim"])

    # --- 3. Search ---
    indices, similarities = _search_corpus(query_emb, embeddings, top_k, min_similarity)
    print(f"  {len(indices)} Treffer (Similarity: {similarities[0]:.4f} – {similarities[-1]:.4f})")

    if len(indices) == 0:
        print("  ⚠ Keine Treffer — leerer AgentState.")
        return AgentState(query=query)

    # --- 4. Load matched posts ---
    posts = _load_matched_posts(indices, similarities, post_ids, posts_df)

    # Log dominant categories
    cat_counts = pd.Series([p["category"] for p in posts]).value_counts()
    top_cats = ", ".join(f"{k} ({v})" for k, v in cat_counts.head(5).items())
    print(f"  Top-Kategorien: {top_cats}")

    # --- 5. Create state + run pipeline ---
    state = AgentState(
        query=query,
        posts=posts,
        limit=top_k,
        max_nudges=3,
    )

    if output_dir is None:
        output_dir = PROJECT_ROOT / "results"

    coordinator = Coordinator(max_revisions=3, verbose=True)
    state = coordinator.run(state, on_phase=on_phase, output_dir=output_dir)

    return state
