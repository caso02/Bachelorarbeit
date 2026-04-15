"""Tests for src/evaluate.py — MRL truncation."""

import numpy as np
import pytest

from src.evaluate import truncate_mrl


class TestTruncateMrl:
    def test_reduces_dimensions(self):
        embeddings = np.random.randn(10, 3072)
        result = truncate_mrl(embeddings, 768)
        assert result.shape == (10, 768)

    def test_l2_normalized(self):
        embeddings = np.random.randn(5, 1024)
        result = truncate_mrl(embeddings, 256)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_noop_when_already_small(self):
        """When target_dim >= D, truncate_mrl returns embeddings unchanged."""
        embeddings = np.random.randn(5, 128)
        result = truncate_mrl(embeddings, 128)
        assert result.shape == (5, 128)
        np.testing.assert_array_equal(result, embeddings)

    def test_larger_target_returns_full(self):
        embeddings = np.random.randn(3, 64)
        result = truncate_mrl(embeddings, 256)
        # Should return original dimensions (not pad)
        assert result.shape[1] == 64
