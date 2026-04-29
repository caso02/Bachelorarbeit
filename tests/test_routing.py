"""Tests for src/routing.py — topic routing via prototype similarity."""

import numpy as np
import pandas as pd
import pytest

from src.routing import route_posts


class TestRoutePosts:
    def test_assigns_correct_top_topic(self):
        """Posts should be assigned to the nearest prototype."""
        # 3 prototypes in 4-dim space
        prototypes = {
            "tech":   np.array([1.0, 0.0, 0.0, 0.0]),
            "sports": np.array([0.0, 1.0, 0.0, 0.0]),
            "health": np.array([0.0, 0.0, 1.0, 0.0]),
        }
        # Posts clearly aligned with each prototype
        posts = np.array([
            [0.9, 0.1, 0.0, 0.0],  # → tech
            [0.1, 0.9, 0.0, 0.0],  # → sports
            [0.0, 0.1, 0.9, 0.0],  # → health
        ])

        df = route_posts(posts, prototypes, top_n=2)

        assert df.loc[0, "topic_1"] == "tech"
        assert df.loc[1, "topic_1"] == "sports"
        assert df.loc[2, "topic_1"] == "health"

    def test_returns_top_n_columns(self):
        prototypes = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
        }
        posts = np.array([[0.7, 0.3]])

        df = route_posts(posts, prototypes, top_n=2)

        assert "topic_1" in df.columns
        assert "score_1" in df.columns
        assert "topic_2" in df.columns
        assert "score_2" in df.columns

    def test_scores_are_between_minus1_and_1(self):
        prototypes = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
        }
        posts = np.random.randn(10, 3)

        df = route_posts(posts, prototypes, top_n=1)

        scores = df["score_1"].values
        assert all(-1.0 <= s <= 1.0 + 1e-6 for s in scores)

    def test_route_travel_posts(self):
        """Travel posts should route to the travel prototype, not other categories."""
        prototypes = {
            "tech":          np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "sports":        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "health":        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "entertainment": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            "environment":   np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            "finance":       np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "travel":        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            "social":        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        }
        # Posts clearly aligned with travel
        posts = np.array([
            [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.0],  # backpacking → travel
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.8, 0.1],  # road trip → travel
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0],  # digital nomad → travel
        ])

        df = route_posts(posts, prototypes, top_n=2)

        assert df.loc[0, "topic_1"] == "travel"
        assert df.loc[1, "topic_1"] == "travel"
        assert df.loc[2, "topic_1"] == "travel"

    def test_route_social_posts(self):
        """Social posts should route to the social prototype, not other categories."""
        prototypes = {
            "tech":          np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "sports":        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "health":        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "entertainment": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            "environment":   np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            "finance":       np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "travel":        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            "social":        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        }
        # Posts clearly aligned with social
        posts = np.array([
            [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.9],  # family boundaries → social
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95],  # friendships → social
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.8],   # relationship advice → social
        ])

        df = route_posts(posts, prototypes, top_n=2)

        assert df.loc[0, "topic_1"] == "social"
        assert df.loc[1, "topic_1"] == "social"
        assert df.loc[2, "topic_1"] == "social"
