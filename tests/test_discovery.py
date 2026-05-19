"""Tests for src/discovery.py — corpus-based semantic search."""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.discovery import _search_corpus, _load_matched_posts


# ---------------------------------------------------------------------------
# _search_corpus tests
# ---------------------------------------------------------------------------

class TestSearchCorpus:
    def test_returns_top_k_indices(self):
        """Only top_k results returned even when more exist."""
        np.random.seed(42)
        corpus = np.random.randn(100, 8).astype(np.float32)
        query = np.random.randn(1, 8).astype(np.float32)

        indices, sims = _search_corpus(query, corpus, top_k=10)

        assert len(indices) == 10
        assert len(sims) == 10

    def test_min_similarity_filter(self):
        """Posts below min_similarity are excluded."""
        # Corpus: 5 posts. Post 0 is identical to query (sim≈1), rest orthogonal (sim≈0)
        query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        corpus = np.array([
            [1.0, 0.0, 0.0, 0.0],  # sim ≈ 1.0
            [0.0, 1.0, 0.0, 0.0],  # sim ≈ 0.0
            [0.0, 0.0, 1.0, 0.0],  # sim ≈ 0.0
            [0.5, 0.5, 0.0, 0.0],  # sim ≈ 0.71
            [0.0, 0.0, 0.0, 1.0],  # sim ≈ 0.0
        ], dtype=np.float32)

        indices, sims = _search_corpus(query, corpus, top_k=10, min_similarity=0.5)

        assert len(indices) == 2  # Only post 0 (sim≈1.0) and post 3 (sim≈0.71)
        assert 0 in indices
        assert 3 in indices

    def test_sorted_descending(self):
        """Results are sorted by similarity descending."""
        np.random.seed(42)
        corpus = np.random.randn(50, 4).astype(np.float32)
        query = np.random.randn(1, 4).astype(np.float32)

        _, sims = _search_corpus(query, corpus, top_k=20)

        for i in range(len(sims) - 1):
            assert sims[i] >= sims[i + 1]


# ---------------------------------------------------------------------------
# _load_matched_posts tests
# ---------------------------------------------------------------------------

class TestLoadMatchedPosts:
    def test_loads_category_metadata(self):
        """Returned posts contain category_1 as 'category' field."""
        import pandas as pd

        post_ids = ["a", "b", "c"]
        posts_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "text_clean": ["text a", "text b", "text c"],
            "subreddit": ["sub1", "sub2", "sub3"],
            "category_1": ["tech", "health", "finance"],
        })
        indices = np.array([0, 2])
        sims = np.array([0.9, 0.7])

        posts = _load_matched_posts(indices, sims, post_ids, posts_df)

        assert len(posts) == 2
        categories = {p["category"] for p in posts}
        assert "tech" in categories
        assert "finance" in categories

    def test_includes_query_similarity(self):
        """Posts include query_similarity field, not score-as-similarity."""
        import pandas as pd

        post_ids = ["x"]
        posts_df = pd.DataFrame({
            "id": ["x"],
            "text_clean": ["some text"],
            "subreddit": ["testsub"],
            "category_1": ["tech"],
        })
        indices = np.array([0])
        sims = np.array([0.85])

        posts = _load_matched_posts(indices, sims, post_ids, posts_df)

        assert posts[0]["score"] == 0
        assert abs(posts[0]["query_similarity"] - 0.85) < 0.01


# ---------------------------------------------------------------------------
# discover integration (mocked)
# ---------------------------------------------------------------------------

class TestDiscoverValidation:
    def test_empty_query_raises(self):
        """Empty query string raises ValueError."""
        from src.discovery import discover
        with pytest.raises(ValueError, match="leer"):
            discover("", corpus_path="/nonexistent")

    def test_whitespace_query_raises(self):
        """Whitespace-only query raises ValueError."""
        from src.discovery import discover
        with pytest.raises(ValueError, match="leer"):
            discover("   ", corpus_path="/nonexistent")
